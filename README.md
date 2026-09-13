# Nexus

Real-time voice-driven visual assistant for the visually impaired. Runs
continuously on a laptop webcam and microphone, listens for the wake word
"Hey Nexus," and answers spoken questions about whatever the camera
currently sees -- object identification, counting, reading printed text
aloud, and identifying Indian currency notes. See `PROJECT_STATUS.md` for
the full build history and key engineering decisions, and
`DrishtiAI_Project_Documentation.md` for the original project proposal this
was built from (see the note at the top of that file for naming history).

## Setup (run these steps in order)

1. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Download the wake-word model** (not included in this repo, ~40MB):
   - Go to https://alphacephei.com/vosk/models
   - Download the **vosk-model-small-en-us** package
   - Unzip it and place the resulting folder in this same directory, named
     exactly `vosk-model-small-en-us` (so `vosk-model-small-en-us/` sits next
     to `vision_saathi.py`)

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
  this") -- also OCR-based: looks for a number matching a real Indian
  denomination (10/20/50/100/200/500/2000) among the recognized text.
  **Known limitation**: this can false-positive on any object with a
  matching number printed on it for an unrelated reason (confirmed by
  testing -- a sunscreen box with "SPF 50" on it was misidentified as a
  50 rupee note). It also depends on the denomination numeral being
  clearly, fully in frame.

## Other scripts

- `voice_vqa_test.py` -- same voice Q&A flow as the main app, but with the
  automatic object-detection alerts turned off, useful for testing just the
  question-answering without interruptions. **Uses the older BLIP-VQA path,
  not moondream2** -- see "About the fine-tuned BLIP model" below.
- `vqa_test.py` -- type a question instead of speaking it (no mic/wake-word
  needed at all). **Also uses the older BLIP-VQA path.**
- `wake_word.py` -- standalone wake-word detection test, isolated from
  everything else.
- `question_recorder.py` -- standalone Whisper transcription test, isolated
  from everything else.
- `diagnose_ollama_moondream.py` -- standalone test of the moondream2/Ollama
  Q&A path alone (captures one webcam frame, asks it a fixed set of sample
  questions, prints the answers), no mic/wake-word/camera-loop involved.
- `eval_test.py` -- the evaluation harness used to produce the accuracy
  numbers in `PROJECT_STATUS.md`. Mirrors `vision_saathi.py`'s actual
  routing logic (kept in sync by hand, not imported, since importing the
  main file would start its live camera loop) against a saved test image
  and a list of questions passed on the command line. Usage:
  `python3 eval_test.py <image_path> "question 1" "question 2" ...`

## Requirements

Python 3.12, a webcam, and a microphone. Developed and tested on macOS
(Apple Silicon). Your OS will likely prompt for camera and microphone
permission the first time you run it -- allow both, or the app can't work.
**This currently runs on a laptop only, not a phone** -- see Limitations in
`PROJECT_STATUS.md`.

## About the fine-tuned BLIP model (research history)

An earlier version of this app answered questions with a BLIP-VQA model
fine-tuned on a VizWiz subset (see `PROJECT_STATUS.md` for the methodology
and measured accuracy improvement: 22.6% baseline -> 25.2% fine-tuned on a
500-question held-out set). That experiment is real, completed work and
its result stands, but it is **not what the live app runs today** -- it was
replaced with moondream2 (via Ollama) because BLIP-VQA can only produce
short, one-word answers by design, no matter how it's fine-tuned, whereas
moondream2 gives full descriptive sentences. `vqa_test.py` and
`voice_vqa_test.py` still use the BLIP/`transformers` path if anyone wants
to compare the two approaches directly.

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
  `python3` -- use `./run.sh` instead of `python3 vision_saathi.py`
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
  see the accuracy breakdown in `PROJECT_STATUS.md` for how often this
  actually happens.
