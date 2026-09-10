# VisionSaathi

Real-time voice-driven visual assistant for the visually impaired. Camera-based
object alerts, wake-word activation ("Hey Nexus"), spoken question answering
about whatever the camera sees. See `PROJECT_STATUS.md` and
`DrishtiAI_Project_Documentation.md` for full project background.

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

## Other scripts

- `voice_vqa_test.py` -- same voice Q&A flow as above, but with the
  automatic object-detection alerts turned off, useful for testing just the
  question-answering without interruptions.
- `vqa_test.py` -- type a question instead of speaking it (no mic/wake-word
  needed at all), useful for quickly testing the VQA model alone.
- `wake_word.py` -- standalone wake-word detection test, isolated from
  everything else.
- `question_recorder.py` -- standalone Whisper transcription test, isolated
  from everything else.
- `diagnose_ollama_moondream.py` -- standalone test of the moondream2/Ollama
  Q&A path alone (captures one webcam frame, asks it a fixed set of sample
  questions, prints the answers), no mic/wake-word/camera-loop involved.

## Requirements

Python 3.12, a webcam, and a microphone. Developed and tested on macOS
(Apple Silicon); should also run on other platforms, though the
`torch.backends.mps` GPU acceleration only applies on Apple Silicon --
elsewhere it automatically falls back to CPU. Your OS will likely prompt
for camera and microphone permission the first time you run it -- allow
both, or the app can't work.

## About the fine-tuned BLIP model (research history)

An earlier version of this app answered questions with a BLIP-VQA model
fine-tuned on a VizWiz subset (see `PROJECT_STATUS.md` Phase 7 for the
methodology and measured accuracy improvement). That fine-tuning experiment
and its results still stand as a real, completed piece of research work --
they just aren't what the live app runs today. It was replaced with
moondream2 (via Ollama) because BLIP-VQA can only produce short, one-word
answers by design, no matter how it's fine-tuned, whereas moondream2 gives
full descriptive sentences -- a better fit for an assistant meant to
describe a scene in detail, not just label it.

The fine-tuned checkpoint itself isn't used by `vision_saathi.py` anymore,
but `vqa_test.py` and `voice_vqa_test.py` still use the BLIP/`transformers`
path if you want to compare the two approaches directly.

## Troubleshooting

- **No sound at all**: check your system volume isn't muted, and try
  `say "test"` (macOS) in a plain terminal to confirm your OS's
  text-to-speech works at all, independent of this app.
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
- **Answers come back oddly short, or empty**: moondream2 can silently
  return nothing for very short/direct questions (confirmed during testing,
  especially for counting questions like "how many..."). The app already
  rephrases every question into a more descriptive prompt before sending it
  to the model to reduce this, but it isn't foolproof for every phrasing.
