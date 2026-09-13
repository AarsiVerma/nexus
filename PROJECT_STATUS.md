# Nexus — Project Status

Real-time visual assistant for visually impaired users, named after its
wake word ("Hey Nexus"). This file is the accurate source of truth for
what's actually been **built and verified working**, the key engineering
decisions and why, and the project's actual current limitations -- if
anything elsewhere (old notes, git history, an earlier draft) conflicts
with this file, this file is correct.

## Hardware constraints (important — shapes every model choice below)

- MacBook Air, Apple **M2**, **8GB RAM**, chronically low on free disk space
  (fluctuated between ~150MB and ~6GB free over the course of development).
- Larger vision-language models (BLIP-2, LLaVA) do **not** fit comfortably
  here — too much RAM/disk. moondream2 (1.9B params, run quantized via
  Ollama, ~1.7GB) was the model that actually fit reliably; see decision
  log below for the failed attempts before landing on it.
- MPS (Apple GPU) is available via PyTorch and is used where it helps
  (Whisper, YOLO, the now-unused BLIP path).

## Files in this directory

- `vision_saathi.py` — **the main integrated app.** Camera + YOLO alerts,
  wake word, Whisper transcription, moondream2 (via Ollama) Q&A, EasyOCR
  text/currency reading, and intent-based routing between them all run
  together in one process/threads. Run via `./run.sh` (see README.md for
  why not plain `python3 vision_saathi.py`).
- `eval_test.py` — evaluation harness mirroring `vision_saathi.py`'s
  routing logic against saved test images, used to produce the accuracy
  numbers below.
- `diagnose_ollama_moondream.py` — standalone moondream2/Ollama test, no
  mic/wake-word/camera-loop.
- `wake_word.py` — standalone wake-word test script (kept for
  reference/testing in isolation; not needed to run the main app).
- `question_recorder.py` / `vqa_test.py` / `voice_vqa_test.py` — standalone
  test scripts. `vqa_test.py` and `voice_vqa_test.py` still use the
  original **BLIP-VQA** path (not moondream2) -- kept for direct
  before/after comparison, not needed for normal use.
- `vosk-model-small-en-us/` — offline wake-word/speech model directory (~40MB,
  downloaded from alphacephei.com/vosk/models). Required by `vision_saathi.py`
  and `wake_word.py`.
- `yolov8n.pt` — YOLOv8 nano weights for object detection.
- `run.sh` — runs the main app with the correct Python interpreter (see
  decision log entry on why this is needed).

## Build status

- **Environment setup**: done. No venv — packages installed globally on
  system Python 3.12 (`/Library/Frameworks/Python.framework/Versions/3.12`).
- **Camera + YOLO object detection**: done and verified working. Automatic
  spoken alerts ("person ahead," etc.) are implemented but currently
  **disabled by default** (`AUTO_ALERTS_ENABLED = False`) -- they were
  found to interrupt/talk over the Q&A flow too often during testing. YOLO
  detection itself still runs continuously regardless, since counting
  relies on it.
- **Wake word**: done and verified working. Wake word is "Hey Nexus," not
  "Hey Nexo" — see decision log for why.
- **Speech-to-text (Whisper)**: done and verified working.
- **Visual question answering**: done via moondream2 (see decision log for
  why, not the originally-planned BLIP-2/LLaVA).
- **OCR-based text reading and Indian currency identification**: done,
  see decision log and Known Limitations below for how currency ID works
  and its false-positive risk.
- **YOLO-based counting**: done, for a fixed set of tracked categories.
- **Full integration**: done. All pieces run together in `vision_saathi.py`;
  extended live testing sessions completed without the wake-word/audio
  issues that were previously blocking this (see decision log).
- **Evaluation**: done in a limited form -- see "Evaluation results" below.
  Sample size is small (13 questions); this is a real limitation, not a
  comprehensive benchmark.
- **Paper/slides**: in progress outside this repo.

## Key decisions and why (don't redo these debates)

1. **Wake word is "Hey Nexus", not "Hey Nexo".** "Nexo" is not a real
   English word and does not exist in Vosk's dictionary at any model size
   (confirmed via direct test — Vosk logs "word missing in vocabulary:
   'nexo'"). Dictionary-based ASR (Vosk, and realistically any similar
   offline engine) can never recognize it, no matter how the grammar/matching
   is tuned. "Nexus" is phonetically almost identical and IS in Vosk's
   dictionary, so that's the actual word the system listens for. The
   project was later renamed to Nexus to match this rather than maintain a
   separate app name and wake word.

2. **Picovoice Porcupine was dropped in favor of Vosk.** Picovoice console
   signup could not be completed (persistent "enter a valid company email"
   error on both personal and college email). Vosk was chosen as a
   zero-account, fully offline replacement. Grammar-constrained
   `KaldiRecognizer` (a restricted word list, not free dictation) is used
   for reliability.

3. **Whisper model size: "tiny", not "base".** Downgraded from "base" to
   "tiny" specifically to reduce RAM footprint after repeated out-of-memory
   crashes. Quality tradeoff is acceptable for short spoken questions, but
   is a real source of transcription errors (confirmed: "describe" was
   misheard as "subscribe" more than once in testing).

4. **Switch from BLIP-VQA to moondream2 (via Ollama) for question
   answering.** Originally used `Salesforce/blip-vqa-base`, fine-tuned on
   VizWiz -- an abandoned approach, not part of the presented project (see
   "Abandoned approach" section below). Even fine-tuned, BLIP-VQA only
   produces short factual answers ("phone," "blue") by architectural
   design -- fine-tuning cannot change that output format. Switched to
   moondream2, a vision-language model built for descriptive answers.
   - **First attempt (raw HF transformers, fp32) OOM-crashed** — a 1.9B
     model at full precision is ~7.6GB, too much for this machine even
     alone.
   - **Second attempt (fp16) also OOM-crashed**, including on a tiny
     224x224 test image, ruling out image size as the cause — the model
     itself doesn't fit at that precision on this hardware.
   - **Working solution: Ollama**, which serves a quantized (~1.7GB)
     version of moondream2 as a local background service. The app talks to
     it over a local connection via the `ollama` Python client instead of
     loading the model in-process.

5. **Prompt engineering findings for moondream2** (confirmed by repeated
   testing, not guesses):
   - Stacking multiple instructions (brevity + vocabulary rules + focus
     hints) into one prompt measurably made answers *worse*, not better.
     The closest-to-raw prompt gave the richest, most accurate answers.
   - Bare "What is X?" style questions reliably fail (return empty, or
     default to describing an unrelated background object), while the same
     question phrased "Can you tell me what is X?" reliably succeeds. The
     app rephrases short questions into this form automatically.
   - Ollama's server-side prompt caching could reuse a stale answer from a
     similar-looking previous frame instead of processing the current one
     -- fixed with a per-question uniqueness marker in the prompt.
   - Deterministic (temperature 0) decoding plus a garbled/foreign-language
     transcription (a genuine Whisper hallucination during testing) once
     triggered a real repetition-loop failure (the model got stuck
     repeating one word hundreds of times). Fixed with a non-ASCII input
     filter (reject non-English transcriptions before they reach the
     model) and a generous `num_predict` cap as a safety net.

6. **Counting is answered from YOLO's live detections, not moondream2.**
   moondream2 was found to be unreliable at counting questions specifically
   (silently returns empty answers). Since YOLO already tracks object
   instances every second for the (currently disabled) alert system, "how
   many X" questions are answered by counting YOLO's current detections for
   a fixed set of tracked categories instead of asking the language model
   to state a number in words. Anything outside that fixed category list
   falls through to moondream2, inheriting its counting unreliability.

7. **Text reading and currency identification use EasyOCR, not
   moondream2.** A general vision-language model isn't reliable at
   precisely transcribing printed text. EasyOCR (English + Hindi) reads
   text directly instead. Currency identification extracts a number
   matching a known Indian denomination from the OCR'd text.
   **Known limitation, confirmed by testing**: this can false-positive on
   any object with a matching number printed on it for an unrelated reason
   (a sunscreen box with "SPF 50" was misidentified as a 50 rupee note).

8. **TTS switched from `pyttsx3` to macOS's native `say` command.**
   `pyttsx3`'s macOS driver could report success (`runAndWait()` returning
   with no exception) while producing no audible sound at all — a silent
   failure mode with nothing for the app to catch or detect, confirmed
   repeatedly during testing. `say` is simpler and was already confirmed
   reliable (used early on to debug system audio) with no equivalent
   silent-failure behavior observed.
   - All speech is still serialized through one queue and one worker
     thread, both to keep utterances from overlapping and (previously) to
     satisfy `pyttsx3`'s same-thread requirement, which no longer applies
     with `say` but the queue is still useful for ordering.
   - `speak_and_wait()` blocks until "I'm listening" has actually finished
     playing before recording starts, since starting the mic immediately
     let the tail end of that phrase bleed into the recorded question.

9. **Single shared microphone stream, not two competing ones.**
   Originally, the wake-word listener kept one mic stream open for the
   whole session while `record_question()` opened and closed a brand new
   separate stream every time a question was recorded. **Confirmed bug**:
   after enough of those open/close cycles, the long-running wake-word
   stream would silently stop receiving audio at all -- "Hey Nexus" simply
   stopped responding partway into a session, with no error. Fixed by
   using one persistent shared stream for the whole app, feeding two
   internal queues (one for wake-word detection, one for question
   recording) from a single audio callback.

10. **Camera capture resolution capped at 960x540**, YOLO runs on a
    resized 640px copy once per second (not every frame) -- memory/CPU
    reduction measures from early debugging.

## Abandoned approach: BLIP-VQA fine-tuning

**Not part of the presented project.** Noted here only so it isn't a
surprise if BLIP-related code turns up in `vqa_test.py`/`voice_vqa_test.py`
or the git history. Early development fine-tuned `blip-vqa-base` on a
VizWiz subset before switching to moondream2 -- it was dropped because
BLIP-VQA only produces short, one-word answers by design regardless of
fine-tuning, which doesn't fit an assistant meant to describe scenes in
detail. If asked directly: yes, this was tried; no, it isn't part of what's
being presented, because the output format itself was the wrong fit, not
because the experiment failed to run.

## Evaluation results (deployed system: moondream2 + OCR routing)

13 live test questions across 3 scenes (a general scene, a held object with
text on it, and a real currency note), scored by manually checking each
answer against the actual photo. See `eval_test.py` for the harness.

| Category | Result |
|---|---|
| General visual Q&A (describe scene, presence/absence, worn items) | 5/6 correct, 1 partial |
| Object identification (what's in hand, reading a label) | 1/2 correct, 1 partial |
| Currency identification | 0/3 correct (always correctly detected "this is a note," never correctly extracted the denomination in this test run) |
| **Overall** | **6/13 correct (46%), 2/13 partial, 5/13 failed** |

**Confirmed failure patterns** (not random noise -- each has a specific,
understood cause):
1. Some short/direct questions ("Is there a person in the picture?", "What
   color is the curtain?") return empty answers even with the
   short-question rephrasing fix in place.
2. OCR text reading is often incomplete -- catches some of the real text
   on an object, misses the rest, occasionally misreads individual
   characters (e.g. "SPF" as "SPE").
3. Currency identification's false-positive risk (see decision #7) and its
   dependence on the denomination numeral being fully, clearly in frame.

**Important context**: 13 questions is a small sample from a single live
session, not a rigorous benchmark -- treat this as a directional,
honestly-reported result, not a definitive accuracy claim. General visual
Q&A and object identification performed well (~75-92%); currency
identification is the clearly weakest, least mature feature.

## Known limitations (current, honest state -- read before demoing or writing this up)

- **Not phone-deployable.** Requires a laptop, a locally-running background
  service (Ollama), and several GB of installed ML libraries. A real
  deployment would need either a much smaller on-device model or a
  phone-as-camera-plus-backend architecture -- neither is implemented.
- **Currency identification can false-positive** on unrelated numbers
  printed on non-currency objects, and only reliably works for a single
  note, fully in frame, with the denomination numeral unobstructed. Cannot
  count or total multiple notes.
- **Counting only works for YOLO's fixed tracked categories** (see decision
  #6) -- cannot count arbitrary objects (fingers, bricks, etc.).
- **English input only for voice.** OCR supports Hindi text reading, but
  spoken interaction (wake word, question, answer) is English-only.
- **No baseline comparison actually run** against existing tools (Seeing
  AI, Lookout) on the same inputs -- only discussed narratively.
- **Small evaluation sample** (13 questions) -- not statistically
  rigorous, reported as a directional result only.
- **Continuous camera use has privacy implications** for bystanders that
  aren't addressed by the current design.
- **Whisper "tiny" makes real transcription errors** (confirmed:
  "describe" misheard as "subscribe"), which can cause a correctly-phrased
  question to still fail downstream.

## Future work

- Multi-note currency counting (detecting and summing several notes at
  once) -- a materially harder problem than single-note identification,
  not attempted yet.
- Mobile/on-device deployment.
- Larger, more systematic evaluation set; direct comparison against
  existing assistive tools on the same inputs.
- Hindi voice input/output (currently text-only via OCR).

## Team contribution table

Equal split by ownership of a distinct, non-overlapping part of the actual
codebase (not a precise effort audit) -- each person should study and be
able to defend their own section, since a viva may ask individually.

| Member | Area of Contribution | Key Deliverables | % |
|---|---|---|---|
| Member 1 | Perception & Counting | YOLO object detection integration, alert system, YOLO-based counting logic | 20% |
| Member 2 | Voice Input Pipeline | Wake word detection (Vosk), speech-to-text (Whisper), microphone architecture & bug fixes | 20% |
| Member 3 | Visual Question Answering (Core) | moondream2/Ollama integration, prompt engineering, model selection & debugging | 20% |
| Member 4 | OCR & Currency Identification | Text reading (EasyOCR), Indian currency denomination detection | 20% |
| Member 5 | TTS, System Integration & Evaluation | Speech output, intent-based routing architecture, evaluation harness & results, fine-tuning experiment | 20% |
| **Total** | | | **100%** |

## Study plan by member (what to read before the viva)

**Member 1 — Perception & Counting**
1. `vision_saathi.py`: the YOLO loading line, the main camera loop's
   detection block (`model(small, verbose=False)`), `ALERT_OBJECTS`,
   `COUNTABLE_OBJECTS`, `count_known_objects()`.
2. Decisions #6 and #10 above, plus the "Camera + YOLO object detection"
   and "YOLO-based counting" bullets under Build Status.
3. Likely questions: How does YOLO detect objects? Why run detection once
   a second instead of every frame? Why are alerts disabled by default?
   Why does counting use YOLO instead of the language model? What happens
   if asked to count something YOLO doesn't track?

**Member 2 — Voice Input Pipeline**
1. `vision_saathi.py`: `wake_word_listener()`, `record_question()`, the
   shared audio stream block (`_shared_audio_callback`,
   `_shared_audio_stream`), the non-ASCII check in `wake_word_handler()`.
2. Decisions #1, #2, #3, #9, and the repetition-loop bullet under #5.
3. Likely questions: Why "Nexus" not "Nexo"? Why Whisper "tiny"? What was
   the microphone bug and how was it fixed? What happened when Whisper
   mistranscribed audio, and how is that prevented now?

**Member 3 — Visual Question Answering (Core)**
1. `vision_saathi.py`: `make_descriptive_prompt()`, the `ollama.chat()`
   call inside `answer_question()`.
2. Decisions #4 and #5 in full -- the densest section, budget the most
   study time here.
3. Likely questions: Why not BLIP-2/LLaVA? What happened with moondream2
   at full precision? How does Ollama solve that? What did you learn about
   prompting this model? What's the caching bug and how did you find it?

**Member 4 — OCR & Currency Identification**
1. `vision_saathi.py`: `read_text_aloud()`, `identify_currency()`,
   `READ_TRIGGERS`, `CURRENCY_TRIGGERS`, `INDIAN_DENOMINATIONS`.
2. Decision #7 and the currency row in Evaluation Results.
3. Likely questions: Why OCR instead of asking the vision model to read
   text? How does currency identification actually work? What's the
   false-positive bug you found, and why does it happen?

**Member 5 — TTS, System Integration & Evaluation**
1. `vision_saathi.py`: `_tts_worker()`, `speak()`/`speak_and_wait()`, and
   the full if/elif routing chain in `answer_question()` (ties all 4 other
   members' parts together -- worth understanding at a high level even
   outside your own section).
2. `eval_test.py`, decision #8, the Evaluation Results table, and the
   Abandoned Approach note.
3. Likely questions: Why drop pyttsx3? What's the overall system
   architecture / how does a question get routed? How was the system
   evaluated, and what did you find? What happened with the BLIP
   fine-tuning experiment, and why isn't it used?
