# Concurrent ANE Utility Model: Use Cases & Architecture Impact

## Overview

Running a ~4B LUT4-palettized model on Apple Neural Engine at ~25-35 tok/s, concurrently with the main MLX chat model on GPU. Zero GPU contention.

---

## Part 1: Existing Tasks That Move to ANE (17 Call Sites)

### 1.1 Pre-Generation Pipeline (Currently 10-22s Blocking GPU)

These run sequentially before the user sees any response tokens, all competing for `mlx_generation_lock`:

#### A. Quick Search Check
- **File:** `python/prompt_analysis.py`, `quick_search_check()` (line 48)
- **Called from:** `server.py` line 7632
- **What it does:** 5-token LLM call ("SEARCH" or "KNOWLEDGE")
- **Current behavior:** Acquires `mlx_generation_lock.priority()` for ~1s
- **ANE benefit:** ~0.2s async, runs while GPU prepares prompt

#### B. Prompt Analysis (Biggest Delay)
- **File:** `python/prompt_analysis.py`, `analyze_prompt()` (line 98)
- **Called from:** `server.py`, `_run_classification()` line 7475
- **What it does:** 300-token LLM call extracting domains, entities, intent, sub-questions, complexity signals
- **Current behavior:** Acquires lock for **4-14 seconds**. Single largest pre-generation delay.
- **ANE benefit:** 8-12s on ANE but runs async — GPU is never blocked

#### C. PubMed Query Optimization
- **File:** `python/query_optimizer.py`, `optimize_pubmed_query()` (line 243)
- **Called from:** `server.py` line 7679
- **What it does:** 40-token call optimizing search terms
- **ANE benefit:** Runs parallel with analysis completing

#### D. Multi-Query Expansion
- **File:** `python/query_optimizer.py`, `optimize_pubmed_queries()` (line 154)
- **Called from:** `server.py` line 7802
- **What it does:** 120-token call generating 2-4 queries from sub-questions
- **ANE benefit:** Fires immediately after analysis, parallel with searches

#### E. Search Result Scoring
- **File:** `python/search_scorer.py`, `score_results()` (line 40)
- **Called from:** `server.py` line 7944
- **What it does:** 80-token call rating each result 1-10 for relevance
- **ANE benefit:** Runs parallel with search results arriving

#### F. Evidence Extraction
- **File:** `python/entity_extractor.py`, `extract_evidence()` (line 37)
- **Called from:** `server.py` line 7954
- **What it does:** 400-token call parsing study_type, sample_size, p_value from abstracts
- **ANE benefit:** Runs parallel with scoring

#### G. Skeleton Decomposition
- **File:** `python/prompt_analysis.py`, `run_full_decomposition()` (line 320)
- **Called from:** `server.py` line 8351
- **What it does:** 600-token call for complex queries
- **ANE benefit:** Runs on ANE while GPU streams the response

### 1.2 Post-Generation Background Tasks (WorkerQueue, Interruptible)

These run via `_bg_generate()` (`server.py` lines 302-378) which acquires/releases the GPU lock **one token at a time** and aborts if a priority request arrives:

#### H. Conversation Tagging
- **File:** `python/conversation_tagger.py`, `extract_topics()` (line 46)
- **Called from:** `server.py` lines 7001, 7426, 9088 (TaskPriority.HIGH)
- **What it does:** 200-token call extracting topic tags from conversation
- **Current behavior:** `_bg_generate()` acquires lock one token at a time, releases between tokens, aborts if priority arrives. Takes 2-5s.
- **ANE benefit:** Runs freely on ANE with zero lock contention, completes in ~6s at 30 tok/s. No interruption needed.

#### I. Objective Direction Update
- **File:** `python/objective_tracker.py`, `update_objectives()` (line 27-105)
- **Called from:** `server.py`, WorkerQueue TaskPriority.HIGH (line 7003, 9088)
- **What it does:** 80-token LLM call updating the one-sentence research direction
- **Current behavior:** `_bg_generate()` with same lock-per-token pattern
- **ANE benefit:** 80 tokens at 30 tok/s = ~2.5s, runs concurrently with tagging

#### J. Mindmap Generation
- **File:** `python/mindmap_generator.py`, `generate_mindmap_md()` (line 89-157)
- **Called from:** `server.py`, WorkerQueue TaskPriority.NORMAL (line 7007)
- **What it does:** 1000-token LLM call converting response to hierarchical markdown
- **Current behavior:** `_bg_generate()`, takes 10-30s with lock yielding
- **ANE benefit:** 1000 tokens at 30 tok/s = ~33s, but runs without any GPU contention. Currently this task is often preempted before completing.

#### K. Skeleton Section Evaluation
- **File:** `server.py`, `_evaluate_sections_llm()` (line 6627-6683)
- **What it does:** 600-token LLM call evaluating each section's quality after skeleton synthesis
- **ANE benefit:** Could run on ANE while the main model generates the next section

#### L. Term Definition Lookup
- **File:** `server.py` (line 3421-3426)
- **What it does:** 100-token LLM call for glossary term definitions
- **ANE benefit:** Trivial task, instant on ANE

### 1.3 Deferred/Idle Tasks

#### M. Background Entity Extraction Worker
- **File:** `server.py`, `_extraction_worker()` (line 5813-5851) and `_interruptible_extract()` (line 5683)
- **What it does:** Processes library document chunks via utility model for knowledge graph entities. Uses `_bg_generate()` with lock-per-token yielding.
- **Current behavior:** Only runs when user is idle for 60+ seconds. Aborts immediately when chat activity resumes. Saves checkpoints per-chunk.
- **ANE benefit:** **Massive improvement.** Could run continuously on ANE without any impact on chat. The entire interruptible/checkpoint machinery becomes unnecessary.

#### N. Library Document Classification (Ontology)
- **File:** `python/search_classifier.py`, `classify_document()` (line 368-411)
- **Called from:** `server.py`, ingestion pipeline and `_backfill_ontology_labels()` (line 5768)
- **What it does:** 150-token LLM call classifying library documents into domain ontology
- **ANE benefit:** Runs during ingestion without blocking chat

#### O. Library Document Summarization
- **File:** `server.py` (line 6014-6016)
- **What it does:** LLM call generating one-line summary from first 2 chunks during ingestion
- **ANE benefit:** Same as N

### 1.4 Layer Analyzer (Calibration)

#### P. Response Segment Classification
- **File:** `python/layer_analyzer.py`, `classify_response_with_llm()` (line 588-687), `_classify_with_llm_sentences()` (line 690-803)
- **What it does:** PER-SENTENCE classification of the probe response (~50 sentences x 10-token calls = ~500 LLM calls). Each call asks "which prompt segment does this sentence answer?"
- **Current behavior:** Uses the **main loaded model** directly via `stream_generate()`. Most wasteful usage -- a 27B+ model doing trivial 1-digit classification tasks.
- **ANE benefit:** A 4B model is perfect for this. The classification task is trivial (pick a number 0-5).

#### Q. LLM Phrase Detection (Fallback)
- **File:** `python/layer_analyzer.py`, `_classify_with_llm_phrases()` (line 806-849)
- **What it does:** 150-token LLM call asking for first words of each section
- **ANE benefit:** Same as P

---

## Part 2: New Capabilities Enabled by Concurrent ANE Inference

### 2.1 Real-Time Input Analysis While User Types

**Currently impossible because the GPU is occupied or the utility model would need the lock.**

- **Keystroke-debounced classification:** As the user types, the ANE model runs `analyze_prompt()` on the current partial input every 500ms. By the time the user hits Send, classification is already done.
- **Live entity extraction:** Show entity chips (like "semaglutide", "HbA1c") in the input area as the user types, confirming the system understands the domain.
- **Predictive source routing:** Show which sources will be searched (PubMed icon lights up, FDA badge appears) before the user submits.
- **Typing complexity indicator:** Show the complexity score updating in real-time, and whether skeleton mode will trigger.

### 2.2 Parallel Response Quality Evaluation (During Streaming)

**Currently impossible because the GPU is busy generating tokens.**

- **Live coherence scoring:** While the main model streams tokens, the ANE model periodically evaluates the accumulated response for coherence, redundancy, and factual consistency.
- **Real-time fact-checking:** ANE model cross-references claims in the streaming response against the search results already in the system prompt, flagging unsupported statements as they appear.
- **Hallucination early warning:** ANE model detects when the response diverges from the provided evidence (PubMed abstracts, library chunks), triggering a visual warning before the full response completes.

### 2.3 Speculative Pre-Computation

- **Pre-fetch search results:** While the user types, ANE classifies the query and starts PubMed/web searches speculatively. If the user sends the message unchanged, results are already cached.
- **Skeleton pre-generation:** For queries ANE predicts will be skeleton-eligible, start decomposition before submission.
- **RAG pre-ranking:** ANE can pre-score library chunks for the predicted query, so RCS re-ranking is instant at submit time.

### 2.4 Concurrent Post-Generation Pipeline

**Currently, post-gen tasks (tagging, objectives, mindmap) run sequentially via WorkerQueue with lock-per-token yielding. If the user sends a new message, all pending tasks are cancelled (`worker_queue.cancel_pending()` at line 7494).**

With ANE:
- **All post-gen tasks run simultaneously** on ANE: tagging + objectives + mindmap can all execute at once since they are independent.
- **No cancellation needed:** ANE tasks don't block the GPU, so the user can send a new message immediately and post-gen tasks from the previous response complete in the background.
- **Instant turnaround:** The 10-30s mindmap generation delay disappears (becomes invisible since it doesn't compete with chat).

### 2.5 Multi-Round Tool Call Optimization

- **File:** `server.py`, `stream_response()` (line 8495, the `while tool_round <= max_tool_rounds` loop)
- **Current behavior:** When the main model emits `<tool_call>` mid-stream, the server pauses token streaming, executes the search, then resumes generation (up to 3 rounds).
- **ANE enhancement:** While the main model is paused waiting for search results, ANE can:
  - Pre-score the incoming search results for relevance
  - Extract evidence from abstracts
  - Prepare optimized context injection
  - Classify whether a second search round will be needed

### 2.6 Continuous Entity Extraction (No Idle Requirement)

- **Current:** Entity extraction for library documents (`_extraction_worker`) only runs after 60s of user inactivity and checkpoints/aborts on any activity.
- **ANE:** Entity extraction runs continuously on ANE. The entire `_IDLE_THRESHOLD`, `_last_chat_activity`, checkpoint-per-chunk, and abort-on-busy machinery becomes unnecessary. Documents get fully processed minutes after upload instead of hours.

### 2.7 Real-Time Search Classifier A/B Evaluation

- **File:** `python/search_classifier.py`, `classify_prompt_simple()` (line 314-343)
- **Currently:** The simple classifier exists for A/B evaluation but is never run concurrently with the ontology classifier because both need the GPU.
- **ANE:** Run both classifiers simultaneously and compare results in real-time for continuous quality monitoring.

### 2.8 Live Document Processing During Chat

- While chatting about a topic, ANE can automatically classify newly ingested library documents, generate summaries, and extract entities -- all without any impact on the ongoing conversation.

---

## Part 3: Architecture Improvements

### 3.1 Eliminate PriorityLock for Utility Tasks

**Current architecture problem (`server.py` lines 148-206):**

The `PriorityLock` (`mlx_generation_lock`) serializes ALL Metal operations. Both the main chat model and the utility model share it. The priority mechanism helps (chat wins over background), but it still means:
1. Pre-gen utility calls (classification, query opt) block until the lock is free
2. Post-gen utility calls (`_bg_generate`) acquire/release per-token
3. If a chat request arrives, background tasks abort and lose work

**ANE fix:** All utility model calls bypass `mlx_generation_lock` entirely. The lock is only needed for the main chat model's Metal operations. This eliminates:
- `PriorityLock._bg_event.wait()` stalls for background tasks
- `_bg_generate()`'s token-by-token lock dance (lines 302-378)
- `worker_queue.cancel_pending()` at the start of each request (line 7494)
- The `_request_counter` staleness check mechanism (line 118)

### 3.2 Flatten the Pre-Generation Pipeline

**Current sequential pipeline in `stream_response()` (lines 7488-7960):**
```
User sends message
  -> is_direct_answer() [<1ms, regex]
  -> quick_search_check() [~1s, GPU lock]
  -> _run_classification() / analyze_prompt() [4-14s, GPU lock]
  -> derive_search_queries() [0ms, deterministic]
  -> optimize_pubmed_query() [~1s, GPU lock]
  -> optimize_pubmed_queries() [~2s, GPU lock]
  -> [PubMed/Web searches run in parallel via asyncio.gather]
  -> score_results() [~1s, HTTP to utility model]
  -> extract_evidence() [~2s, HTTP to utility model]
  -> [Start streaming first token]
Total pre-gen delay: 10-22 seconds before the first token.
```

**ANE-optimized pipeline:**
```
User sends message (or while typing)
  -> is_direct_answer() [<1ms, regex, unchanged]
  -> ANE: analyze_prompt() [starts immediately, ~8-12s]
  -> ANE: quick_search_check() [parallel with analysis, ~1s]
  -> GPU: Start warming up / preparing formatted prompt
  -> [ANE analysis completes]
  -> ANE: optimize_pubmed_queries() [~3s]
  -> [PubMed/Web searches fire immediately]
  -> ANE: score_results() [parallel with searches returning]
  -> ANE: extract_evidence() [parallel with scoring]
  -> GPU: Start streaming first token [GPU was free the entire time]
New pre-gen delay: 0s for GPU (it was never blocked).
```

### 3.3 Remove _bg_generate() Token-by-Token Lock Dance

**File:** `server.py`, `_bg_generate()` (lines 302-378)

This function is a complex workaround for the single-GPU constraint. It:
1. Acquires the background lock
2. Generates one token
3. Releases the lock
4. Checks if priority is waiting
5. If so, aborts and returns None
6. Otherwise, sleeps 10ms and loops

With ANE, replace with a simple:
```python
async def ane_generate(prompt, max_tokens, temperature):
    return await ane_model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
```
No lock, no yielding, no abort, no checkpointing.

### 3.4 Simplify WorkerQueue to Parallel Dispatch

**File:** `server.py`, `WorkerQueue` class (lines 221-297)

Currently processes one task at a time because all tasks share the GPU lock. With ANE:
- Tagging, objectives, and mindmap can run as independent `asyncio.create_task()` calls
- No priority ordering needed (ANE handles all simultaneously)
- `cancel_pending()` at request start becomes unnecessary
- The entire `WorkerQueue` class could be replaced with `asyncio.TaskGroup`

### 3.5 Eliminate Entity Extraction Idle Machinery

**File:** `server.py`, `_extraction_worker()` (lines 5813-5870)

Remove:
- `_last_chat_activity` timestamp tracking (line 121)
- `_IDLE_THRESHOLD` constant (line 123)
- `_system_busy()` helper (lines 5825-5830)
- 5-second polling loop waiting for idle (lines 5832-5838)
- Per-chunk abort-on-activity checks (lines 5658-5680)
- Checkpoint save/restore on abort (lines 5663-5700)

Replace with simple sequential chunk processing on ANE -- no interruption logic needed.

### 3.6 Layer Analyzer Decoupling

**File:** `python/layer_analyzer.py`

Currently `classify_response_with_llm()` uses `stream_generate()` on the main model. With ANE:
- Classification runs on ANE (4B model is sufficient for "pick a number 0-5")
- Response generation stays on the main model
- The two are completely decoupled

---

## Impact Summary

| Category | Current State | With ANE |
|----------|--------------|----------|
| **Time to first token** | 10-22s (GPU blocked by utility tasks) | ~0s GPU wait (utility runs on ANE) |
| **Post-gen task completion** | Often cancelled, sequential, 10-30s | Always completes, parallel, invisible |
| **Entity extraction** | Hours (idle-only, checkpoint/abort) | Minutes (continuous, no interruption) |
| **Layer analysis classification** | ~25s using main model | ~15s on ANE, fully decoupled |
| **Code complexity** | PriorityLock, _bg_generate, WorkerQueue, idle detection | Simple async function calls |
| **User experience during typing** | Static input box | Live classification, entity chips, source prediction |
| **Mid-stream quality checks** | Impossible | Continuous coherence/hallucination monitoring |
