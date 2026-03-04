// main.cpp — ane-lm: Apple Neural Engine LLM inference tool
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <csignal>
#include <string>
#include <vector>
#include <utility>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/select.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <ane_lm/common.h>
#include "utils.h"
#include "generate.h"
#include "core/model_loader.h"
#include "core/ane_runtime.h"

// ObjC autorelease pool via C runtime API
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

static volatile sig_atomic_t g_serve_stop = 0;

static void serve_signal_handler(int) {
    g_serve_stop = 1;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s generate --model <path> [--prompt <text>] [options]\n", prog);
    fprintf(stderr, "  %s chat --model <path> [options]\n", prog);
    fprintf(stderr, "  %s serve --model <path> --socket <path> [options]\n", prog);
    fprintf(stderr, "  %s convert --model <path>\n", prog);
    fprintf(stderr, "\nSubcommands:\n");
    fprintf(stderr, "  generate    Single-shot text generation\n");
    fprintf(stderr, "  chat        Interactive multi-turn chat\n");
    fprintf(stderr, "  serve       Persistent daemon (Unix socket, JSON-lines protocol)\n");
    fprintf(stderr, "  convert     Convert model weights from BF16 to FP16\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --model <path>    Path to model directory (required)\n");
    fprintf(stderr, "  --prompt <text>   Input prompt (generate only, default: \"Hello\")\n");
    fprintf(stderr, "  --socket <path>   Unix socket path (serve only, required)\n");
    fprintf(stderr, "  --max-tokens N    Max tokens per response (default: unlimited)\n");
    fprintf(stderr, "  --temp T          Temperature (default: 0.6)\n");
    fprintf(stderr, "  --repeat-penalty P Repetition penalty (default: 1.2, 1.0=off)\n");
    fprintf(stderr, "  --enable-thinking Enable thinking/reasoning mode\n");
    fprintf(stderr, "  --no-ane-cache    Disable persistent ANE compile cache\n");
    fprintf(stderr, "  -v, --verbose     Show detailed initialization info\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s generate --model /path/to/Qwen3.5-0.8B --prompt \"Hello\" --max-tokens 50\n", prog);
    fprintf(stderr, "  %s chat --model /path/to/Qwen3.5-0.8B\n", prog);
    fprintf(stderr, "  %s serve --model /path/to/Qwen3.5-0.8B --socket /tmp/ane.sock\n", prog);
}

struct Args {
    const char* model_dir = nullptr;
    const char* prompt = "Hello";
    const char* socket_path = nullptr;
    float temperature = 0.6f;
    int max_tokens = 0;
    float repetition_penalty = 1.2f;
    bool ane_cache = true;
    bool enable_thinking = false;
};

static Args parse_args(int argc, char* argv[], int start) {
    Args args;
    for (int i = start; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            args.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            args.repetition_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            args.socket_path = argv[++i];
        } else if (strcmp(argv[i], "--enable-thinking") == 0) {
            args.enable_thinking = true;
        } else if (strcmp(argv[i], "--no-ane-cache") == 0) {
            args.ane_cache = false;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            g_verbose = true;
        }
    }
    return args;
}

static int cmd_generate(LLMModel& model, Tokenizer& tokenizer, const Args& args) {
    LOG("Prompt: \"%s\"\n", args.prompt);

    SamplingParams sampling;
    sampling.temperature = args.temperature;
    sampling.repetition_penalty = args.repetition_penalty;

    GenerationResponse last{};
    bool first = true;
    stream_generate(model, tokenizer, std::string(args.prompt),
        args.max_tokens, args.enable_thinking, sampling,
        [&](const GenerationResponse& r) {
            if (r.token == -1) {
                last = r;
                return;
            }
            if (!r.text.empty()) {
                if (first) { fprintf(stderr, "==========\n"); first = false; }
                fprintf(stderr, "%s", r.text.c_str());
            }
            last = r;
        });

    fprintf(stderr, "\n==========\n");
    fprintf(stderr, "Prompt: %d tokens, %.3f tokens-per-sec\n",
            last.prompt_tokens, last.prompt_tps);
    fprintf(stderr, "Generation: %d tokens, %.3f tokens-per-sec\n",
            last.generation_tokens, last.generation_tps);
    return 0;
}

static int cmd_chat(LLMModel& model, Tokenizer& tokenizer, const Args& args) {
    std::vector<std::pair<std::string, std::string>> messages;
    char buf[4096];

    while (true) {
        fprintf(stderr, ">>> ");
        if (!fgets(buf, sizeof(buf), stdin)) {
            // EOF (Ctrl-D)
            fprintf(stderr, "\n");
            break;
        }

        // Strip trailing newline
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

        // Skip empty input
        if (buf[0] == '\0') continue;

        // Exit commands
        if (strcmp(buf, "/bye") == 0 || strcmp(buf, "/exit") == 0) break;

        // Add user message
        messages.push_back({"user", std::string(buf)});

        // Reset model state and generate with full history
        model.reset();

        SamplingParams sampling;
        sampling.temperature = args.temperature;
        sampling.repetition_penalty = args.repetition_penalty;

        std::string assistant_text;
        GenerationResponse last{};
        stream_generate(model, tokenizer, messages,
            args.max_tokens, args.enable_thinking, sampling,
            [&](const GenerationResponse& r) {
                if (r.token == -1) {
                    last = r;
                    return;
                }
                if (!r.text.empty()) {
                    fprintf(stderr, "%s", r.text.c_str());
                    assistant_text += r.text;
                }
                last = r;
            });

        fprintf(stderr, "\n");

        // Add assistant response to history
        messages.push_back({"assistant", assistant_text});

        fprintf(stderr, "[%d prompt tokens, %.1f t/s | %d gen tokens, %.1f t/s]\n\n",
                last.prompt_tokens, last.prompt_tps,
                last.generation_tokens, last.generation_tps);
    }

    return 0;
}

static int cmd_serve(LLMModel& model, Tokenizer& tokenizer, const Args& args) {
    using json = nlohmann::json;

    signal(SIGTERM, serve_signal_handler);
    signal(SIGINT, serve_signal_handler);
    signal(SIGPIPE, SIG_IGN);  // Prevent client disconnect from killing daemon

    // Create Unix domain socket
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "[serve] Failed to create socket\n");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, args.socket_path, sizeof(addr.sun_path) - 1);

    unlink(args.socket_path);  // Remove stale socket
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[serve] Failed to bind to %s\n", args.socket_path);
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 5) < 0) {
        fprintf(stderr, "[serve] Failed to listen\n");
        close(server_fd);
        unlink(args.socket_path);
        return 1;
    }

    fprintf(stderr, "[serve] Listening on %s\n", args.socket_path);

    while (!g_serve_stop) {
        // Use select() with 1s timeout for graceful shutdown
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        struct timeval tv = {1, 0};
        int ready = select(server_fd + 1, &rfds, nullptr, nullptr, &tv);
        if (ready <= 0) continue;

        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;

        // Set socket timeouts to prevent hangs on broken clients
        struct timeval sock_tv = {10, 0};  // 10s
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &sock_tv, sizeof(sock_tv));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &sock_tv, sizeof(sock_tv));

        // Read request line (until \n)
        std::string request_line;
        char ch;
        while (true) {
            ssize_t n = read(client_fd, &ch, 1);
            if (n <= 0 || ch == '\n') break;
            request_line += ch;
        }

        if (request_line.empty()) {
            close(client_fd);
            continue;
        }

        // Per-request autorelease pool
        void* req_pool = objc_autoreleasePoolPush();

        json resp;
        std::string req_id;

        try {
            json req = json::parse(request_line);
            req_id = req.value("id", "");
            std::string prompt = req.value("prompt", "Hello");
            int max_tokens = req.value("max_tokens", 200);
            float temperature = req.value("temperature", 0.1f);
            float rep_penalty = req.value("repetition_penalty", 1.0f);

            // Reset model state for independent request
            model.reset();

            SamplingParams sampling;
            sampling.temperature = temperature;
            sampling.repetition_penalty = rep_penalty;

            std::string full_text;
            GenerationResponse last{};

            stream_generate(model, tokenizer, prompt,
                max_tokens, false, sampling,
                [&](const GenerationResponse& r) {
                    if (r.token == -1) {
                        last = r;
                        return;
                    }
                    if (!r.text.empty()) {
                        full_text += r.text;
                    }
                    last = r;
                });

            resp["id"] = req_id;
            resp["text"] = full_text;
            resp["prompt_tokens"] = last.prompt_tokens;
            resp["gen_tokens"] = last.generation_tokens;
            resp["prompt_tps"] = last.prompt_tps;
            resp["gen_tps"] = last.generation_tps;
            resp["error"] = nullptr;

            fprintf(stderr, "[serve] req=%s gen=%d tokens @ %.1f t/s\n",
                    req_id.c_str(), last.generation_tokens, last.generation_tps);

        } catch (const std::exception& e) {
            resp["id"] = req_id;
            resp["text"] = "";
            resp["error"] = std::string(e.what());
            fprintf(stderr, "[serve] req=%s error: %s\n", req_id.c_str(), e.what());
        }

        objc_autoreleasePoolPop(req_pool);

        // Write response (check for errors but never crash)
        std::string resp_line = resp.dump() + "\n";
        ssize_t written = write(client_fd, resp_line.c_str(), resp_line.size());
        if (written < 0) {
            fprintf(stderr, "[serve] req=%s write failed: %s\n",
                    req_id.c_str(), strerror(errno));
        }
        close(client_fd);
    }

    close(server_fd);
    unlink(args.socket_path);
    fprintf(stderr, "[serve] Stopped.\n");
    return 0;
}

static int cmd_convert(const Args& args) {
    std::string model_dir = args.model_dir;

    // Discover all safetensors files (single-file or sharded) and convert them.
    auto weights = ModelWeights::open(model_dir);
    if (!weights) {
        fprintf(stderr, "Error: failed to load model weights in %s\n", model_dir.c_str());
        return 1;
    }

    std::string output_dir = model_dir + "/ane_weights";

    Timer timer;
    int written = weights->write_ane_blobs(output_dir);
    double elapsed = timer.elapsed_ms();

    if (written < 0) {
        fprintf(stderr, "Error: conversion failed\n");
        return 1;
    }

    fprintf(stderr, "Done in %.1f ms\n", elapsed);
    return 0;
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    srand48(time(nullptr));
    setbuf(stdout, nullptr);

    // Need at least a subcommand
    if (argc < 2) {
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // Check for --help before subcommand
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 0;
    }

    // Determine subcommand
    const char* subcmd = argv[1];
    bool is_generate = (strcmp(subcmd, "generate") == 0);
    bool is_chat = (strcmp(subcmd, "chat") == 0);
    bool is_serve = (strcmp(subcmd, "serve") == 0);
    bool is_convert = (strcmp(subcmd, "convert") == 0);

    bool is_test_quant = (strcmp(subcmd, "test-quantize") == 0);
    bool is_test_lut4 = (strcmp(subcmd, "test-lut4") == 0);

    if (!is_generate && !is_chat && !is_serve && !is_convert && !is_test_quant && !is_test_lut4) {
        fprintf(stderr, "Unknown subcommand: %s\n\n", subcmd);
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    if (is_test_quant) {
        g_verbose = true;
        ane_test_quantized_compile();
        objc_autoreleasePoolPop(pool);
        return 0;
    }

    if (is_test_lut4) {
        g_verbose = true;
        ane_test_lut4_compile();
        objc_autoreleasePoolPop(pool);
        return 0;
    }

    // Parse args after subcommand
    Args args = parse_args(argc, argv, 2);

    if (!args.model_dir) {
        fprintf(stderr, "Error: --model is required\n\n");
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    // convert doesn't need model/tokenizer loading
    if (is_convert) {
        int ret = cmd_convert(args);
        objc_autoreleasePoolPop(pool);
        return ret;
    }

    // serve requires --socket
    if (is_serve && !args.socket_path) {
        fprintf(stderr, "Error: --socket is required for serve subcommand\n\n");
        print_usage(argv[0]);
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    LOG("=== ane-lm: Apple Neural Engine LLM Inference ===\n");
    LOG("Model: %s\n", args.model_dir);
    LOG("Mode: %s\n", is_serve ? "serve" : (is_chat ? "chat" : "generate"));
    LOG("Temperature: %.2f, Max tokens: %d\n", args.temperature, args.max_tokens);
    LOG("ANE compile cache: %s\n", args.ane_cache ? "enabled" : "disabled");

    // Load model + tokenizer
    std::unique_ptr<LLMModel> model;
    Tokenizer tokenizer;
    try {
        auto result = load(args.model_dir, args.ane_cache);
        model = std::move(result.first);
        tokenizer = std::move(result.second);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        objc_autoreleasePoolPop(pool);
        return 1;
    }

    int ret;
    if (is_serve) {
        ret = cmd_serve(*model, tokenizer, args);
    } else if (is_chat) {
        ret = cmd_chat(*model, tokenizer, args);
    } else {
        ret = cmd_generate(*model, tokenizer, args);
    }

    objc_autoreleasePoolPop(pool);
    return ret;
}
