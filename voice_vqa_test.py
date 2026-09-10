import cv2
import time
import json
import threading
import queue
import numpy as np
import sounddevice as sd
import whisper
import torch
import pyttsx3
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from vosk import Model as VoskModel, KaldiRecognizer

print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("Loading VQA model...")
VQA_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VQA_DTYPE = torch.float16 if VQA_DEVICE == "mps" else torch.float32
VQA_MODEL_PATH = "blip_finetuned"
vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL_PATH)
vqa_model = BlipForQuestionAnswering.from_pretrained(
    VQA_MODEL_PATH, torch_dtype=VQA_DTYPE
).to(VQA_DEVICE)

latest_frame = None

speech_queue = queue.Queue()


def _tts_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = speech_queue.get()
        engine.say(text)
        engine.runAndWait()


threading.Thread(target=_tts_worker, daemon=True).start()


def speak(text):
    speech_queue.put(text)


# --- Wake word ("Hey Nexus"), same setup as vision_saathi.py -------------
WAKE_MODEL_PATH = "vosk-model-small-en-us"
WAKE_GRAMMAR = ["hey", "nexus", "[unk]"]
WAKE_COOLDOWN = 3

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
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MAX_QUESTION_DURATION = 15
CHUNK_DURATION = 0.25


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
    return audio.astype(np.float32) / 32768.0


def phrase_answer(question, raw_answer):
    q = question.lower()
    if q.startswith(("how many", "count")):
        return f"There are {raw_answer}."
    if q.startswith("what color") or "colour" in q:
        return f"It looks {raw_answer}."
    if q.startswith(("is ", "are ", "can ", "do ", "does ")):
        return f"{raw_answer.capitalize()}."
    return f"I can see {raw_answer}."


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
        except Exception as e:
            print(f"Error: {e}")


threading.Thread(target=wake_word_listener, daemon=True).start()
threading.Thread(target=wake_word_handler, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
print("Ready. No object-detection alerts here -- say 'Hey Nexus' then ask your question.")
print("Press Q in the camera window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    latest_frame = frame

    cv2.imshow('VoiceVQATest', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
