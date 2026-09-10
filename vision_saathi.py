import cv2
import pyttsx3
import threading
import queue
import json
import os
import numpy as np
import sounddevice as sd
import whisper
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from vosk import Model as VoskModel, KaldiRecognizer
from ultralytics import YOLO
import time

# Load YOLOv8 model (downloads automatically on first run)
model = YOLO('yolov8n.pt')

print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("Loading VQA model...")
VQA_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# float16 halves the model's memory footprint; MPS supports it well, plain
# CPU inference does not, so only use it when running on the GPU.
VQA_DTYPE = torch.float16 if VQA_DEVICE == "mps" else torch.float32
# Fine-tuned on a VizWiz subset (see PROJECT_STATUS.md Phase 7). Falls back
# to the stock model if the fine-tuned checkpoint isn't present locally --
# it's too large for git, so anyone cloning the repo without it still gets
# a working app.
VQA_MODEL_PATH = "blip_finetuned" if os.path.isdir("blip_finetuned") else "Salesforce/blip-vqa-base"
vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL_PATH)
vqa_model = BlipForQuestionAnswering.from_pretrained(
    VQA_MODEL_PATH, torch_dtype=VQA_DTYPE
).to(VQA_DEVICE)

latest_frame = None  # updated every main-loop iteration, read by the wake-word handler

# Setup text to speech

# Objects we care about alerting for
ALERT_OBJECTS = {
    'person': 'Person ahead',
    'chair': 'Chair in the way',
    'dining table': 'Table ahead',
    'bottle': 'Bottle nearby',
    'cup': 'Cup nearby',
    'laptop': 'Laptop on surface',
    'cell phone': 'Phone nearby',
    'book': 'Book nearby',
    'stairs': 'Stairs ahead, be careful',
    'door': 'Door ahead',
    'car': 'Car nearby, be careful',
    'motorcycle': 'Motorcycle nearby, be careful',
    'bicycle': 'Bicycle nearby',
    'dog': 'Dog nearby',
    'cat': 'Cat nearby',
}

AUTO_ALERTS_ENABLED = False  # set True to re-enable automatic "Person ahead" style alerts

last_spoken = {}
consecutive_hits = {}
COOLDOWN = 5  # seconds between repeating same alert
DETECT_INTERVAL = 1.0  # run YOLO at most once per second
CONFIDENCE_THRESHOLD = 0.6
CONFIRM_FRAMES = 2  # require this many consecutive detections before alerting, cuts flicker/false positives

qa_active = threading.Event()  # set while a wake-word question is being recorded/answered


# pyttsx3's macOS speech driver isn't safe to call from multiple threads at
# once (it fatally crashes the whole process, not just the thread), so all
# speech goes through one queue and one worker thread.
speech_queue = queue.Queue()

def _tts_worker():
    # pyttsx3's macOS driver must be initialized and driven from the same
    # thread throughout its life, or speech silently no-ops (no error, no
    # sound) — so the engine is created here, not on the main thread.
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = speech_queue.get()
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

def should_alert(label):
    now = time.time()
    if label not in last_spoken or (now - last_spoken[label]) > COOLDOWN:
        last_spoken[label] = now
        return True
    return False


# --- Wake word ("Hey Nexus") ---------------------------------------------
# "Nexo" isn't a real English word, so it can't be recognized by
# dictionary-based speech recognition (Vosk has no entry for it). "Nexus"
# is phonetically almost identical and is in the dictionary, so that's the
# actual spoken trigger. The recognizer's grammar is restricted to just
# these words, which makes detection far more reliable than free dictation.
WAKE_MODEL_PATH = "vosk-model-small-en-us"
WAKE_GRAMMAR = ["hey", "nexus", "[unk]"]
WAKE_COOLDOWN = 3  # seconds, avoid re-triggering on the same utterance

wake_word_detected = threading.Event()


def wake_word_listener():
    vosk_model = VoskModel(WAKE_MODEL_PATH)
    recognizer = KaldiRecognizer(vosk_model, 16000, json.dumps(WAKE_GRAMMAR))
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(bytes(indata))

    last_trigger = 0
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                            channels=1, callback=callback):
        while True:
            data = audio_queue.get()
            heard = None
            if recognizer.AcceptWaveform(data):
                heard = json.loads(recognizer.Result()).get("text", "")
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                if "nexus" in partial:
                    heard = partial
                    recognizer.Reset()

            if heard and "nexus" in heard and (time.time() - last_trigger) > WAKE_COOLDOWN:
                last_trigger = time.time()
                wake_word_detected.set()


QUESTION_SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500     # RMS amplitude below this counts as silence
SILENCE_DURATION = 1.5      # stop after this many seconds of silence
MAX_QUESTION_DURATION = 15  # hard cap so it never records forever
CHUNK_DURATION = 0.25       # seconds per audio chunk read from the mic


def record_question():
    chunks = []
    silence_start = None
    start_time = time.time()

    stream = sd.InputStream(samplerate=QUESTION_SAMPLE_RATE, channels=1, dtype='int16')
    stream.start()

    while True:
        data, _ = stream.read(int(QUESTION_SAMPLE_RATE * CHUNK_DURATION))
        chunks.append(data.copy())

        rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_DURATION:
                break
        else:
            silence_start = None

        if time.time() - start_time > MAX_QUESTION_DURATION:
            break

    stream.stop()
    stream.close()

    audio = np.concatenate(chunks, axis=0).flatten()
    return audio.astype(np.float32) / 32768.0  # Whisper expects float32 in [-1, 1]


def phrase_answer(question, raw_answer):
    # BLIP VQA always returns a short factual answer (e.g. "phone", "blue"),
    # never a full sentence -- this wraps it to sound like a natural spoken
    # response instead of a flat template repeated every time.
    q = question.lower().strip()
    answer = raw_answer.strip()

    if q.startswith(("how many", "count")):
        return f"There are {answer} of them."
    if q.startswith("what color") or "colour" in q:
        return f"It looks {answer} in color."
    if q.startswith(("where", "where's", "where is")):
        return f"It looks like it's {answer}."
    if q.startswith(("who", "who's", "who is")):
        return f"That looks like {answer}."
    if q.startswith(("is ", "are ", "can ", "do ", "does ", "did ", "was ", "were ")):
        if answer == "yes":
            return "Yes, that's right."
        if answer == "no":
            return "No, it isn't."
        return f"{answer.capitalize()}."
    if q.startswith(("what is", "what's", "what are", "what does")):
        return f"That looks like {answer}."
    return f"It looks like {answer}."


def answer_question(question):
    if latest_frame is None:
        return "I can't see anything right now."

    image = Image.fromarray(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB))
    inputs = vqa_processor(image, question, return_tensors="pt").to(VQA_DEVICE, VQA_DTYPE)
    out = vqa_model.generate(**inputs, max_new_tokens=30)
    raw_answer = vqa_processor.decode(out[0], skip_special_tokens=True)
    return phrase_answer(question, raw_answer)


def wake_word_handler():
    while True:
        wake_word_detected.wait()
        wake_word_detected.clear()
        qa_active.set()
        try:
            print("Wake word detected: Hey Nexus")
            speak("I'm listening")

            audio = record_question()
            result = whisper_model.transcribe(audio, language="en", fp16=False)
            question = result["text"].strip()
            print(f"Question heard: {question}")

            if not question:
                speak("Sorry, I didn't catch that.")
                continue

            answer = answer_question(question)
            print(f"Answer: {answer}")
            speak(answer)
        finally:
            qa_active.clear()


threading.Thread(target=wake_word_listener, daemon=True).start()
threading.Thread(target=wake_word_handler, daemon=True).start()

# Open webcam. The default capture resolution (often 1920x1080) burns extra
# memory and CPU on every frame for no benefit here, so cap it lower.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
print("VisionSaathi is running... Press Q to quit.")
speak("VisionSaathi is ready")

last_detect_time = 0
last_boxes = []  # persisted detections drawn between YOLO runs

while True:
    ret, frame = cap.read()
    if not ret:
        break

    latest_frame = frame

    now = time.time()
    if now - last_detect_time >= DETECT_INTERVAL:
        last_detect_time = now

        # Run detection on a smaller copy for speed
        small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
        scale_x = frame.shape[1] / small.shape[1]
        scale_y = frame.shape[0] / small.shape[0]

        results = model(small, verbose=False)
        last_boxes = []
        seen_labels = set()

        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)

                if confidence > CONFIDENCE_THRESHOLD and label in ALERT_OBJECTS:
                    seen_labels.add(label)
                    consecutive_hits[label] = consecutive_hits.get(label, 0) + 1

                    if (AUTO_ALERTS_ENABLED
                            and not qa_active.is_set()
                            and consecutive_hits[label] >= CONFIRM_FRAMES
                            and should_alert(label)):
                        print(f"Detected: {label} ({confidence:.0%})")
                        speak(ALERT_OBJECTS[label])

                    x1, y1, x2, y2 = box.xyxy[0]
                    last_boxes.append((
                        int(x1 * scale_x), int(y1 * scale_y),
                        int(x2 * scale_x), int(y2 * scale_y),
                        label, confidence,
                    ))

        # reset the streak for any alert object that vanished this frame
        for label in list(consecutive_hits):
            if label not in seen_labels:
                consecutive_hits[label] = 0

    for x1, y1, x2, y2, label, confidence in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(frame, f"{label} {confidence:.0%}",
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 200, 100), 2)

    cv2.imshow('VisionSaathi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("VisionSaathi stopped.")