import cv2
import pyttsx3
#import subprocess
import threading
import queue
import json
import re
import os
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
VQA_FRAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexus_frame.jpg")
try:
    ollama.list()
except Exception as e:
    raise SystemExit(
        "Can't reach the Ollama service. "
        f"Make sure Ollama is running and that `ollama pull {VQA_MODEL}` "
        f"has been completed before starting this app.\n"
        f"Original error: {e}"
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

# Currency questions also route to OCR rather than moondream -- identifying
# a note is fundamentally reading the denomination number printed on it
# (in multiple spots, on every Indian note), not visually classifying it by
# color/design, which a small general VLM isn't reliable at.
CURRENCY_TRIGGERS = (
    'how much money', 'how much is this', 'what note', 'which note',
    'what currency', 'rupee', 'rupees',
)
INDIAN_DENOMINATIONS = {'10', '20', '50', '100', '200', '500', '2000'}

# Text every genuine Indian note carries (English + Hindi) but an unrelated
# object with a matching number on it (e.g. "SPF 50") won't -- used as a
# second signal in identify_currency() below to cut false positives.
CURRENCY_KEYWORDS = ('reserve bank', 'rbi', 'भारतीय रिज़र्व बैंक', 'रिज़र्व बैंक')

# Every Indian note also prints its amount spelled out in words (Hindi and
# English), e.g. "पाँच सौ रुपये" / "FIVE HUNDRED RUPEES" on a 500 note --
# confirmed by testing that OCR sometimes catches this instead of (or as
# well as) the plain numeral. Each entry lists token sets (both spelling
# variants where relevant); all tokens in a set must appear in the same
# OCR'd fragment to count as a match for that denomination.
DENOMINATION_WORD_TOKENS = {
    '10': [['दस'], ['ten', 'rupees']],
    '20': [['बीस'], ['twenty', 'rupees']],
    '50': [['पचास'], ['fifty', 'rupees']],
    '100': [['एक', 'सौ'], ['one', 'hundred']],
    '200': [['दो', 'सौ'], ['two', 'hundred']],
    '500': [['पाँच', 'सौ'], ['पांच', 'सौ'], ['five', 'hundred']],
    '2000': [['दो', 'हज़ार'], ['दो', 'हजार'], ['two', 'thousand']],
}


def _word_form_denomination(kept):
    # Lowercased for the English tokens above -- .lower() is a no-op on
    # Devanagari, so the Hindi tokens still match unaffected.
    for text in kept:
        text_lower = text.lower()
        for digits, token_sets in DENOMINATION_WORD_TOKENS.items():
            if any(all(tok in text_lower for tok in tokens) for tokens in token_sets):
                return digits
    return None

AUTO_ALERTS_ENABLED = False  # set True to re-enable automatic "Person ahead" style alerts

last_spoken = {}
consecutive_hits = {}
COOLDOWN = 5  # seconds between repeating same alert
DETECT_INTERVAL = 1.0  # run YOLO at most once per second
CONFIDENCE_THRESHOLD = 0.6
CONFIRM_FRAMES = 2  # require this many consecutive detections before alerting, cuts flicker/false positives

qa_active = threading.Event()  # set while a wake-word question is being recorded/answered


# --- Text to speech -------------------------------------------------------

# --- Text to speech -------------------------------------------------------

speech_queue = queue.Queue()


def _speak_windows(text):
    """Speak one message using a fresh Windows SAPI5 engine."""
    engine = None

    try:
        print(f"[TTS] starting: {text[:60]!r}...", flush=True)

        engine = pyttsx3.init("sapi5")
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        # Make sure the engine has a valid Windows voice.
        voices = engine.getProperty("voices")

        if voices:
            engine.setProperty("voice", voices[0].id)

        engine.say(text)
        engine.runAndWait()

        print("[TTS] finished OK", flush=True)

    except Exception as e:
        print(f"TTS error: {e}", flush=True)

    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

        engine = None


def _tts_worker():
    print("[TTS] Windows SAPI5 worker started", flush=True)

    while True:
        text = speech_queue.get()

        try:
            _speak_windows(text)
        finally:
            speech_queue.task_done()


threading.Thread(target=_tts_worker, daemon=True).start()


def speak(text):
    if text:
        speech_queue.put(str(text))


def speak_and_wait(text):
    speak(text)
    speech_queue.join()

def should_alert(label):
    now = time.time()
    if label not in last_spoken or (now - last_spoken[label]) > COOLDOWN:
        last_spoken[label] = now
        return True
    return False


# --- Shared microphone stream ---------------------------------------------
# Originally, the wake-word listener kept one mic stream open for the whole
# session while record_question() opened and closed a brand new separate
# stream every single time a question was recorded. Confirmed by testing:
# after enough of those open/close cycles, the long-running wake-word stream
# would silently stop receiving audio at all (no crash, no error -- "Hey
# Nexus" just stopped working partway into a session). Two concurrent
# streams to the same mic device apparently isn't reliable over time on this
# hardware. Fix: one persistent stream for the entire app, shared by both
# consumers via two separate queues fed from the same callback.
AUDIO_SAMPLE_RATE = 16000

wake_audio_queue = queue.Queue()
question_audio_queue = queue.Queue()
recording_active = threading.Event()  # only buffer into question_audio_queue while actually recording


def _shared_audio_callback(indata, frames, time_info, status):
    data = bytes(indata)
    wake_audio_queue.put(data)
    if recording_active.is_set():
        question_audio_queue.put(data)


_shared_audio_stream = sd.RawInputStream(
    samplerate=AUDIO_SAMPLE_RATE, blocksize=8000, dtype='int16',
    channels=1, callback=_shared_audio_callback,
)
_shared_audio_stream.start()


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
    recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE, json.dumps(WAKE_GRAMMAR))

    last_trigger = 0
    while True:
        data = wake_audio_queue.get()
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


SILENCE_THRESHOLD = 500     # RMS amplitude below this counts as silence
SILENCE_DURATION = 1.5      # stop after this many seconds of silence
MAX_QUESTION_DURATION = 15  # hard cap so it never records forever


def record_question():
    # Drop any stale audio that arrived while we weren't recording, so the
    # first chunk we process is actually from now, not a backlog.
    while not question_audio_queue.empty():
        question_audio_queue.get_nowait()

    recording_active.set()
    chunks = []
    silence_start = None
    start_time = time.time()

    try:
        while True:
            try:
                data = question_audio_queue.get(timeout=MAX_QUESTION_DURATION)
            except queue.Empty:
                break
            chunk = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
            chunks.append(chunk)

            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            if rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    break
            else:
                silence_start = None

            if time.time() - start_time > MAX_QUESTION_DURATION:
                break
    finally:
        recording_active.clear()

    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).flatten()
    return audio.astype(np.float32) / 32768.0  # Whisper expects float32 in [-1, 1]


_question_counter = 0


def make_descriptive_prompt(question):
    # Close-to-raw -- the very first, unconstrained test of this model gave
    # the richest, most detailed answers of any version tried, and every
    # instruction piled on after that (brevity, vocabulary, focus hints)
    # made answers worse, confirmed by repeated testing.
    #
    # One exception, confirmed repeatedly across multiple test sessions:
    # bare "What is in my hand?" reliably fails (defaults to describing a
    # background object), while the *exact same question* phrased as a
    # request -- "Can you tell/describe what is in my hand?" -- reliably
    # succeeds. (A generic "please describe in detail:" prefix was tried
    # first and did NOT fix it -- this specific "can you tell me" phrasing
    # is what the logs actually show working, so use that instead of a
    # guess.)
    global _question_counter
    _question_counter += 1
    q = question.strip()
    if not q.endswith(('.', '?', '!')):
        q += '?'
    if not q.lower().startswith(('can you', 'could you', 'would you', 'please')):
        q = f"Can you tell me {q[0].lower()}{q[1:]}"
    return f"[Question #{_question_counter}] {q}"


def read_text_aloud():
    results = ocr_reader.readtext(latest_frame)
    # confidence > 0.25 filters out genuine noise/garbage while keeping more
    # of the real but fainter/smaller text that 0.4 was dropping
    kept = [(bbox, text) for (bbox, text, confidence) in results if confidence > 0.25]
    if not kept:
        return "I couldn't find any readable text there."
    # EasyOCR doesn't guarantee reading order -- sort top-to-bottom, then
    # left-to-right (using each box's top-left corner) so a multi-line
    # label comes out in the order a person would actually read it
    kept.sort(key=lambda item: (item[0][0][1], item[0][0][0]))
    texts = [text for (_, text) in kept]
    return "It says: " + ", ".join(texts)


def identify_currency():
    results = ocr_reader.readtext(latest_frame)
    kept = [text for (_, text, confidence) in results if confidence > 0.25]
    if not kept:
        return "I couldn't find a currency note clearly in view."

    has_bank_text = any(
        keyword in " ".join(kept).lower() for keyword in CURRENCY_KEYWORDS
    )

    # Indian notes print the denomination as a plain number in multiple
    # spots (both corners, the numeral panel, the watermark window) --
    # count how many OCR fragments matched each denomination, since an
    # unrelated object with one matching number on it (confirmed by
    # testing: a sunscreen box's "SPF 50") should only ever produce one hit.
    digit_hits = {}
    for text in kept:
        digits = re.sub(r'[^0-9]', '', text)
        if digits in INDIAN_DENOMINATIONS:
            digit_hits[digits] = digit_hits.get(digits, 0) + 1

    for digits, count in digit_hits.items():
        # Either signal alone is enough: bank text confirms it's a real note
        # even if only one instance of the number is legible, and 2+
        # instances of the number is strong evidence even without bank text.
        if has_bank_text or count >= 2:
            return f"This looks like a {digits} rupee note."

    # The amount spelled out in words (Hindi/English) is on its own strong
    # enough evidence -- an unrelated object saying "पाँच सौ रुपये" by
    # coincidence isn't realistic -- so it doesn't need a numeral to back it
    # up, unlike the single-numeral case above.
    word_digits = _word_form_denomination(kept)
    if word_digits:
        return f"This looks like a {word_digits} rupee note."

    if digit_hits:
        return ("I can see a number that could be a denomination, but "
                "nothing else about it looks like currency -- this might "
                "not actually be a note.")

    # Found text (probably bank name, promise-to-pay line, etc.) but no
    # recognizable denomination number -- read what we did find rather than
    # just failing silently.
    return "I can see a note but couldn't clearly read the amount. It says: " + ", ".join(kept)


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

    if any(trigger in q for trigger in CURRENCY_TRIGGERS):
        return identify_currency()

    if any(trigger in q for trigger in READ_TRIGGERS):
        return read_text_aloud()

    # Was `q.startswith(("how many", "count"))` -- too brittle for real
    # spoken phrasing, which rarely opens with the trigger word ("please
    # count...", "...and count them?", "describe the number of..."),
    # confirmed by testing to silently fall through to moondream (bad at
    # counting) instead of YOLO. Match the whole phrase anywhere in the
    # question instead.
    if re.search(r'\bcount\b|\bhow many\b|\bnumber of\b', q):
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
        # No brevity instruction or temperature override -- confirmed by
        # testing that those made answers worse. num_predict here is purely
        # a safety net, not a quality tweak: with greedy/deterministic
        # decoding, a bad (e.g. garbled/foreign-language) prompt can trigger
        # a genuine repetition loop with no randomness to break out of it,
        # generating unbounded text that would otherwise get queued to
        # speak in full. 200 tokens is well beyond any real answer seen in
        # testing (normal rich answers run 3-4 sentences), so it shouldn't
        # cut off legitimate responses.
        options={"num_predict": 200},
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
            speak_and_wait("I'm listening")

            audio = record_question()
            result = whisper_model.transcribe(audio, language="en", fp16=False)
            question = result["text"].strip()
            print(f"Question heard: {question}")

            # Whisper occasionally hallucinates garbage on unclear/noisy
            # audio -- including, confirmed by testing, other scripts
            # (Korean) mixed with stray symbols. Sending that to the VQA
            # model is worse than useless: it can trigger a genuine
            # repetition-loop failure (deterministic decoding has no
            # randomness to break out of one once started), producing a
            # huge wall of repeated text that then gets queued to speak.
            # This app only ever expects English, so reject anything with
            # non-ASCII characters before it reaches the model at all.
            if not question or not question.isascii():
                speak("Sorry, I didn't catch that.")
                continue

            try:
                answer = answer_question(question)
                print(f"Answer: {answer}")
                speak(answer)
            except Exception as e:
                print(f"VQA error: {e}", flush=True)
                speak("Sorry, I couldn't process that question.")
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
print("Nexus is running... Press Q to quit.")
speak("Nexus is ready")

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

    # Detection boxes are no longer drawn on screen (not needed -- YOLO
    # still runs every second for the counting feature, just not displayed).

    cv2.imshow('Nexus', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Nexus stopped.")