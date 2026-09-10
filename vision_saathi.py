import cv2
import pyttsx3
import threading
import queue
import json
import re
import numpy as np
import sounddevice as sd
import whisper
import ollama
import easyocr
from vosk import Model as VoskModel, KaldiRecognizer
from ultralytics import YOLO
import time

# Load YOLOv8 model (downloads automatically on first run)
model = YOLO('yolov8n.pt')

print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("Checking Ollama / VQA model...")
# Answers questions via moondream2 (quantized, served locally by Ollama) instead
# of BLIP-VQA -- gives full descriptive sentences instead of one-word answers.
# Requires the Ollama background service running (`brew services start ollama`)
# and the model pulled (`ollama pull moondream:v2`).
VQA_MODEL = "moondream:v2"
VQA_FRAME_PATH = "/tmp/vision_saathi_frame.jpg"
try:
    ollama.list()
except Exception as e:
    raise SystemExit(
        "Can't reach the Ollama service. Run `brew services start ollama` "
        f"(and make sure `ollama pull {VQA_MODEL}` has been run) before "
        f"starting this app.\nOriginal error: {e}"
    )

print("Loading OCR reader (first run downloads ~100-200MB)...")
# moondream isn't a dedicated OCR tool and can't reliably transcribe printed
# text (labels, signs, currency) -- EasyOCR handles that instead, including
# Hindi alongside English, since that's a stated goal for Indian-context use.
ocr_reader = easyocr.Reader(['en', 'hi'])

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

# Maps words a user might actually say to the YOLO label that counts them.
# Counting questions are answered by counting YOLO's own detections instead
# of asking moondream to state a number in words -- moondream silently fails
# on counting questions (confirmed by testing), while YOLO already tracks
# exactly these object instances every second for the alert system above.
COUNTABLE_OBJECTS = {
    'person': 'person', 'people': 'person', 'persons': 'person',
    'bottle': 'bottle', 'bottles': 'bottle',
    'cup': 'cup', 'cups': 'cup',
    'chair': 'chair', 'chairs': 'chair',
    'laptop': 'laptop', 'laptops': 'laptop',
    'phone': 'cell phone', 'phones': 'cell phone', 'cellphone': 'cell phone',
    'book': 'book', 'books': 'book',
    'car': 'car', 'cars': 'car',
    'motorcycle': 'motorcycle', 'motorcycles': 'motorcycle',
    'bicycle': 'bicycle', 'bicycles': 'bicycle', 'bike': 'bicycle', 'bikes': 'bicycle',
    'dog': 'dog', 'dogs': 'dog',
    'cat': 'cat', 'cats': 'cat',
    'table': 'dining table', 'tables': 'dining table',
    'door': 'door', 'doors': 'door',
}

# Phrases that mean "read this out loud" rather than "describe/answer" --
# routed to OCR instead of moondream, since a general vision-language model
# isn't reliable at precisely transcribing printed text.
READ_TRIGGERS = ('read', 'says', 'written', 'say', 'text on', 'label say')

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


def make_descriptive_prompt(question):
    # moondream2 (via Ollama) silently returns an empty answer for short,
    # direct questions ("How many people are there?") but answers well when
    # the same question is framed descriptively -- confirmed by testing, see
    # PROJECT_STATUS.md. Wrapping every question this way avoids depending on
    # how the user happens to phrase things out loud. The "one or two
    # sentences, stay focused" instruction curbs a separate tendency to
    # ramble into unrelated scene details instead of answering what was
    # actually asked.
    q = question.strip()
    if not q.endswith(('.', '?', '!')):
        q += '?'
    return (
        "Looking at this image, answer this question in one or two short "
        f"sentences, staying focused only on what was asked: {q}"
    )


def read_text_aloud():
    # confidence > 0.4 filters out low-confidence noise/garbage detections,
    # not genuine but faint text
    results = ocr_reader.readtext(latest_frame)
    texts = [text for (_, text, confidence) in results if confidence > 0.4]
    if not texts:
        return "I couldn't find any readable text there."
    return "It says: " + ", ".join(texts)


def count_known_objects(question):
    words = re.findall(r"[a-z']+", question.lower())
    for word in words:
        if word in COUNTABLE_OBJECTS:
            label = COUNTABLE_OBJECTS[word]
            count = sum(1 for box in last_boxes if box[4] == label)
            if count == 0:
                return f"I don't see any {word} right now."
            return f"I can see {count} {word} right now."
    return None  # not a recognized/countable object -- caller falls back to moondream


def answer_question(question):
    if latest_frame is None:
        return "I can't see anything right now."

    q = question.lower()

    if any(trigger in q for trigger in READ_TRIGGERS):
        return read_text_aloud()

    if q.startswith(("how many", "count")):
        count_answer = count_known_objects(q)
        if count_answer is not None:
            return count_answer
        # not a YOLO-tracked object -- fall through to moondream below,
        # which can still attempt a general descriptive answer

    cv2.imwrite(VQA_FRAME_PATH, latest_frame)
    response = ollama.chat(
        model=VQA_MODEL,
        messages=[{
            "role": "user",
            "content": make_descriptive_prompt(question),
            "images": [VQA_FRAME_PATH],
        }],
        # Hard cap as a safety net -- keeps answers short even if the model
        # doesn't fully follow the "one or two sentences" instruction above.
        options={"num_predict": 60},
    )
    answer = response["message"]["content"].strip()
    if not answer:
        return "I'm not too sure about that one -- could you ask it a different way?"
    return answer


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


# Defined before the wake-word threads start so count_known_objects() always
# has something to read, even if the wake word fires before the first
# detection pass below completes.
last_detect_time = 0
last_boxes = []  # persisted detections drawn between YOLO runs

threading.Thread(target=wake_word_listener, daemon=True).start()
threading.Thread(target=wake_word_handler, daemon=True).start()

# Open webcam. The default capture resolution (often 1920x1080) burns extra
# memory and CPU on every frame for no benefit here, so cap it lower.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
print("VisionSaathi is running... Press Q to quit.")
speak("VisionSaathi is ready")

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