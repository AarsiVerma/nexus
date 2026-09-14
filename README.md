# Nexus

Real-time voice-driven visual assistant for the visually impaired. Runs
continuously on a laptop webcam and microphone, listens for the wake word
"Hey Nexus," and answers spoken questions about whatever the camera
currently sees -- object identification, counting, reading printed text
aloud, and identifying Indian currency notes. See `PROJECT_STATUS.md` for
the full build history, key engineering decisions, and the actual
evaluation results -- that file is the accurate source of truth for what
this project does and doesn't do.

## Setup (run these steps in order)

**Windows and macOS supported** — Windows uses pyttsx3 with SAPI5 for TTS; macOS uses the built-in `say` command. Linux is untested but should work with a TTS backend.

1. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Download the wake-word model** (not included in this repo, ~40MB):
   - Go to https://alphacephei.com/vosk/models
   - Download the **vosk-model-small-en-us** package
   - Unzip it and place the resulting folder in this same directory, named
     exactly `vosk-model-small-en-us` (so `vosk-model-small-en-us/` sits next
     to `nexus.py`)

3. **Install and start Ollama** (runs the question-answering model locally):
   ```
   brew install ollama
   brew services start ollama
   ollama pull moondream:v2
   ```
   The pull is about 1.7GB, one-time only. Ollama needs to be running
   (`brew services start ollama` sets it to start automatically going
   forward) whenever you run the app.

4. **`yolov8n.pt`** (YOLO object detection weights) downloads automatically
   on first run, no action needed.

5. **Run it:**
   ```
   ./run.sh
   ```
   Say "Hey Nexus" to ask a question about what the camera sees. Press `Q`
   in the camera window to quit.

   (If installing Ollama via Homebrew pulled in its own Python and now
   shadows your original one when you type plain `python3`, `run.sh` sidesteps
   that by pointing directly at the correct interpreter -- see Troubleshooting
   below if you'd rather run it manually.)

## What it can actually do (verified working, see PROJECT_STATUS.md for the evaluation)

- **General visual questions** ("what do you see," "what's in front of me,"
  "what am I holding") -- answered by moondream2, a vision-language model
  run locally via Ollama, giving full descriptive sentences rather than
  single-word labels.
- **Counting** for a fixed set of object categories YOLO already tracks
  (person, bottle, cup, chair, laptop, phone, book, car, motorcycle,
  bicycle, dog, cat, table, door) -- answered from YOLO's own live object
  detection counts, not by asking the language model to state a number.
  Counting anything outside this list falls back to moondream2, which is
  known to be unreliable at counting in general.
- **Reading printed text aloud** ("read this," "what does it say") --
  answered via EasyOCR (English + Hindi), not the vision-language model,
  since a general VLM isn't reliable at precisely transcribing text.
- **Indian currency identification** ("how much is this," "what note is
  this") -- also OCR-based: accepts a denomination match (10/20/50/100/
  200/500/2000) only when backed by more than a single bare number, to
  avoid misreading an unrelated object as a note (originally confirmed by
  testing: a sunscreen box's "SPF 50" was misidentified as a 50 rupee
  note). It looks for any of three signals: the number printed more than
  once (real notes print it in multiple spots), bank text ("RESERVE BANK
  OF INDIA" / भारतीय रिज़र्व बैंक), or the amount spelled out in words in
  Hindi or English (e.g. "पाँच सौ रुपये" / "FIVE HUNDRED RUPEES"). **Known
  limitation**: still depends on OCR actually catching one of those
  signals clearly in frame, and a contrived object carrying two matching
  numbers or real bank-style text could still fool it.

## Other scripts

- `voice_vqa_test.py` -- same voice Q&A flow as the main app, but with the
  automatic object-detection alerts turned off, useful for testing just the
  question-answering without interruptions. **Uses an older, abandoned
  BLIP-VQA approach, not moondream2 -- not part of the presented project,
  kept only as a dev artifact.**
- `vqa_test.py` -- type a question instead of speaking it (no mic/wake-word
  needed at all). **Also uses the abandoned BLIP-VQA approach, not part of
  the presented project.**
- `wake_word.py` -- standalone wake-word detection test, isolated from
  everything else.
- `question_recorder.py` -- standalone Whisper transcription test, isolated
  from everything else.
- `diagnose_ollama_moondream.py` -- standalone test of the moondream2/Ollama
  Q&A path alone (captures one webcam frame, asks it a fixed set of sample
  questions, prints the answers), no mic/wake-word/camera-loop involved.
- `eval_test.py` -- an evaluation harness for testing question routing
  against saved photos instead of the live camera (no scored benchmark is
  currently published from it -- see "Sample interactions" in
  `PROJECT_STATUS.md` for real question/answer examples instead). Mirrors
  `nexus.py`'s actual routing logic (kept in sync by hand, not imported,
  since importing the main file would start its live camera loop) against
  a saved test image
  and a list of questions passed on the command line. Usage:
  `python3 eval_test.py <image_path> "question 1" "question 2" ...`

## Requirements

**Windows (tested) and macOS (tested)** — Linux untested but should work
with a TTS backend.

- **Windows**: Uses `pyttsx3` with SAPI5 (included in `requirements.txt`).
  Run with `python nexus.py` or `run.bat`.
- **macOS**: Uses built-in `say` command (no extra install).
  Run with `./run.sh`.
- Both: Python 3.10+, webcam, microphone, Ollama running locally with
  `moondream:v2` pulled.

## Note on an abandoned approach

Early development fine-tuned a BLIP-VQA model instead of using moondream2.
**This was abandoned and is not part of the project being presented** --
mentioned here only so it's not a surprise if you see BLIP-related code in
`vqa_test.py`/`voice_vqa_test.py` or in the git history. It was dropped
because BLIP-VQA only produces short, one-word answers by design, which
doesn't fit an assistant meant to describe scenes in detail. See
`PROJECT_STATUS.md` if you need the specifics of why it was tried and why
it was dropped.

## Troubleshooting

- **No sound at all**: check your system volume isn't muted, and try
  `say "test"` (macOS) in a plain terminal to confirm your OS's
  text-to-speech works at all, independent of this app. (The app itself
  uses this same `say` command for speech, not a Python TTS library --
  we switched away from `pyttsx3` after confirming it could silently
  report success while producing no audible sound at all.)
- **Camera window is black or frozen**: check camera permission was
  granted to your terminal app in your OS's privacy settings.
- **"No module named X" error**: rerun `pip install -r requirements.txt`.
- **"Can't reach the Ollama service" error**: run `brew services start
  ollama`, and confirm the model is pulled with `ollama list` (should show
  `moondream:v2`). If you installed Ollama after already having Python
  packages installed, Homebrew's own Python may now shadow your original
  `python3` -- use `./run.sh` instead of `python3 nexus.py`
  directly, or find your original interpreter with
  `ls /Library/Frameworks/Python.framework/Versions/*/bin/python3` and call
  that explicitly.
- **"Hey Nexus" stops working partway through a session**: this was a real,
  confirmed bug (two competing microphone streams disrupting each other
  over time) and has been fixed by moving to a single shared audio stream.
  If it recurs, that fix should be the first thing checked.
- **Answers come back empty, or oddly short**: moondream2 can silently
  return nothing for certain short/direct question phrasings (confirmed by
  testing -- e.g. "What is in my hand?" alone can fail while "Can you tell
  me what is in my hand?" succeeds). The app rephrases short questions
  automatically to reduce this, but it isn't foolproof for every phrasing --
  see "Confirmed failure patterns" under Sample Interactions in
  `PROJECT_STATUS.md` for known cases.
