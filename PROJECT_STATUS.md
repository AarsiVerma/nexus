# VisionSaathi (formerly "DrishtiAI") — Project Status

Real-time visual assistant for visually impaired users. Original full spec/doc
was pasted at project start (10-phase plan, dataset/paper plan for an ML
research project called DrishtiAI). This file tracks what's actually been
**built and verified working** so far, plus key decisions and open issues.

## Hardware constraints (important — shapes every model choice below)

- MacBook Air, Apple **M2**, **8GB RAM**, only **~2-4GB free disk** at any
  given time (this machine is chronically low on both RAM and disk).
- BLIP-2 / LLaVA (what the original doc recommended) do **not** fit
  comfortably here — too much RAM/disk. Everything below uses lighter
  swaps instead.
- MPS (Apple GPU) is available via PyTorch and is used where it helps.

## Files in this directory

- `vision_saathi.py` — **the main integrated app.** Camera + YOLO alerts,
  wake word, Whisper transcription, and VQA answering all run together in
  one process/threads. This is the one to run: `python3 vision_saathi.py`
- `wake_word.py` — standalone wake-word test script (kept for reference/testing
  in isolation; not needed to run the main app).
- `question_recorder.py` / `vqa_test.py` — standalone test scripts used to
  validate Whisper recording and BLIP VQA answering in isolation before they
  were wired into `vision_saathi.py`. Not needed for normal use, kept for
  debugging.
- `vosk-model-small-en-us/` — offline wake-word/speech model directory (~40MB,
  downloaded from alphacephei.com/vosk/models). Required by both
  `vision_saathi.py` and `wake_word.py`.
- `yolov8n.pt` — YOLOv8 nano weights for object detection.

## Phase status (against the original 10-phase doc)

- **Phase 1 (env setup)**: done. No venv — packages installed globally on
  system Python 3.12 (`/Library/Frameworks/Python.framework/Versions/3.12`).
- **Phase 2 (camera + YOLO alerts)**: done and verified working.
- **Phase 3 (wake word)**: done and verified working, but **wake word is
  "Hey Nexus", not "Hey Nexo"** — see decision log below for why.
- **Phase 4 (Whisper question recording)**: done and verified working.
- **Phase 5 (VLM answering)**: done and verified working for basic Q&A, but
  the **memory-crash issue (see "Current blocker" below) was not yet
  confirmed fixed** — last thing in progress before this handoff.
- **Phase 6 (full integration)**: effectively done — all pieces already run
  together in `vision_saathi.py`. Not yet stress-tested for 5+ minutes
  continuous run per the doc's Phase 6 acceptance criteria.
- **Phases 7-10** (Colab fine-tuning, Indian-context evaluation, paper
  writing, demo prep): **not started.**

## Key decisions and why (don't redo these debates)

1. **Wake word is "Hey Nexus", not "Hey Nexo".** "Nexo" is not a real
   English word and does not exist in Vosk's dictionary at any model size
   (confirmed via direct test — Vosk logs "word missing in vocabulary:
   'nexo'"). Dictionary-based ASR (Vosk, and realistically any similar
   offline engine) can never recognize it, no matter how the grammar/matching
   is tuned. "Nexus" is phonetically almost identical and IS in Vosk's
   dictionary, so that's the actual word the system listens for. If true
   "Nexo" recognition is ever required, the only real fix is training a
   custom acoustic model (e.g. openWakeWord), not tuning Vosk.

2. **Picovoice Porcupine was dropped in favor of Vosk.** The user could not
   complete Picovoice console signup (persistent "enter a valid company
   email" error, rejected personal and college email both — likely a bug or
   they were being routed through Picovoice's enterprise/sales form rather
   than the actual self-serve signup). Vosk was chosen as a zero-account,
   fully offline replacement. Grammar-constrained `KaldiRecognizer` (a
   restricted word list, not free dictation) is used for reliability.

3. **Whisper model size: "tiny", not "base".** Downgraded from "base" to
   "tiny" specifically to reduce RAM footprint after repeated out-of-memory
   crashes. Quality tradeoff is acceptable for short spoken questions.

4. **VLM is BLIP VQA base (`Salesforce/blip-vqa-base`), not BLIP-2 or
   LLaVA.** The original doc's suggested BLIP-2 (~3GB download, ~2.7B
   params) does not fit this machine's disk/RAM budget. Blip-vqa-base is
   ~360M params, ~1.5GB download, and is loaded in **float16 on MPS**
   specifically to halve its memory footprint (float16 is NOT used on CPU —
   only enabled when `torch.backends.mps.is_available()`).
   - BLIP VQA models give short factual answers ("mirror", not a sentence).
     A `phrase_answer()` function in `vision_saathi.py` wraps the raw answer
     into a fuller spoken sentence based on simple question-type heuristics
     (how many / what color / yes-no / generic).

5. **Camera capture resolution capped at 960x540** (`cap.set(...)`) instead
   of the default (often 1920x1080) to cut memory/CPU load — this was one of
   several memory-reduction changes.

6. **YOLO alert loop redesigned to decouple display from inference**: YOLO
   only runs once per second (`DETECT_INTERVAL`) on a resized 640px copy of
   the frame, while the display window refreshes every frame — otherwise the
   video window appeared black/frozen (each YOLO inference on a full-res
   frame took multiple seconds, blocking the display refresh).

7. **All TTS (`pyttsx3`) calls are serialized through one queue + one
   worker thread**, never called concurrently from multiple threads. This
   isn't just cleanliness — calling `engine.runAndWait()` from multiple
   threads concurrently reliably **crashes the entire Python process**
   (macOS's NSSpeechSynthesizer backend is not thread-safe), which is what
   was originally causing the app to silently die a few seconds after
   startup. Do not revert this.

8. **YOLO alerts require 2 consecutive-frame confirmation and are
   suppressed while a wake-word question is being recorded/answered**
   (`qa_active` threading.Event). This fixed two complaints: random/flickery
   false-positive alerts, and alerts audibly interrupting/talking over the
   Q&A flow.

## Current blocker / last unresolved issue

The app was repeatedly crashing silently (no traceback, no crash report —
just a `resource_tracker: leaked semaphore` warning at shutdown, which is
consistent with the OS force-killing the process rather than a Python-level
exception). Diagnosis process:

- Confirmed disk space is very tight (was down to ~2-4GB free at points)
  and RAM is 8GB total.
- Measured actual RSS of loading all 4 models (YOLO + Whisper tiny + Vosk +
  BLIP fp16) in isolation: only ~620MB combined — not by itself excessive.
- Checked system-wide memory via `psutil.virtual_memory()` and Activity
  Monitor: **the real problem is other background processes, not this
  app.** Activity Monitor showed multiple VS Code "Code Helper"
  (Plugin/Renderer) processes still running and consuming **~2.2GB
  combined**, even though the user believed VS Code was closed. Closing all
  VS Code windows does NOT quit the app on macOS — the helper/renderer/
  extension-host processes keep running until VS Code is actually quit
  (Cmd+Q or right-click Dock icon → Quit). Activity Monitor also showed
  **844.5MB of swap already in use**, confirming genuine system-wide memory
  pressure.
- **Fix instructed but not yet confirmed**: fully quit VS Code (Cmd+Q, not
  just closing windows), verify in Activity Monitor that "Code Helper"
  processes actually disappear, *then* re-run `python3 vision_saathi.py`
  and test the full "Hey Nexus" → question → spoken answer flow.

**Next step when resuming**: confirm whether quitting VS Code properly
fixed the crash. If it still crashes with VS Code fully quit and Activity
Monitor showing real headroom, the next thing to check is whether some
*other* app (browser tabs, Creative Cloud background services, etc.) is
still eating memory — Activity Monitor's Memory tab, sorted by Mem column,
is the fastest way to check.

## Known easy wins not yet done

- Adobe Creative Cloud background services (UI Helper, Content Manager,
  Core Service) were seen idling at ~330MB combined in Activity Monitor —
  not necessary unless actively using Adobe apps, could be quit for more
  headroom during testing.
- Phase 6's "run 5+ minutes without crashing" acceptance test hasn't been
  done yet — worth doing once the memory issue is confirmed resolved.
