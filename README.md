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

3. **Everything else downloads automatically on first run:**
   - `yolov8n.pt` (YOLO object detection weights) downloads automatically
   - The VQA model downloads automatically from Hugging Face
     (`Salesforce/blip-vqa-base`) unless a `blip_finetuned/` folder is
     present in this directory, in which case that fine-tuned checkpoint is
     used instead. Most people cloning this repo won't have that folder
     (it's a custom fine-tuned model too large for git) -- that's fine, the
     app automatically falls back to the stock model with no changes needed.

4. **Run it:**
   ```
   python3 vision_saathi.py
   ```
   Say "Hey Nexus" to ask a question about what the camera sees. Press `Q`
   in the camera window to quit.

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

## Requirements

Python 3.12, a webcam, and a microphone. Developed and tested on macOS
(Apple Silicon); should also run on other platforms, though the
`torch.backends.mps` GPU acceleration only applies on Apple Silicon --
elsewhere it automatically falls back to CPU. Your OS will likely prompt
for camera and microphone permission the first time you run it -- allow
both, or the app can't work.

## Using the fine-tuned model instead of the stock one

If you were sent a `blip_finetuned` folder (or a zip of it), unzip it if
needed, then place the folder -- named exactly `blip_finetuned` -- directly
in this project directory, next to `vision_saathi.py`. No code changes
needed; it's detected and used automatically if present.

## Troubleshooting

- **No sound at all**: check your system volume isn't muted, and try
  `say "test"` (macOS) in a plain terminal to confirm your OS's
  text-to-speech works at all, independent of this app.
- **Camera window is black or frozen**: check camera permission was
  granted to your terminal app in your OS's privacy settings.
- **"No module named X" error**: rerun `pip install -r requirements.txt`.
