import sounddevice as sd
import queue
import json
import sys
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-small-en-us"
SAMPLE_RATE = 16000

# "Nexo" isn't a real English word, so it doesn't exist in Vosk's
# dictionary and can never be recognized by a dictionary-based ASR model
# (confirmed: the model logs "word missing in vocabulary: 'nexo'").
# "Nexus" is phonetically almost identical and IS in the dictionary, so
# that's the actual spoken trigger. We also restrict the recognizer's
# grammar to just these words, which makes detection far more reliable
# than free-form dictation.
WAKE_TRIGGERS = ["nexus"]
GRAMMAR = ["hey", "nexus", "[unk]"]

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(GRAMMAR))

audio_queue = queue.Queue()


def callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


def contains_wake_word(text):
    text = text.lower()
    return any(trigger in text for trigger in WAKE_TRIGGERS)


print("Listening for 'Hey Nexo'... (Ctrl+C to stop)")

with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                        channels=1, callback=callback):
    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print(f"Heard: {text}")
                if contains_wake_word(text):
                    print("ACTIVATED")
        else:
            partial = json.loads(recognizer.PartialResult())
            ptext = partial.get("partial", "")
            if ptext and contains_wake_word(ptext):
                print(f"Heard (partial): {ptext}")
                print("ACTIVATED")
                recognizer.Reset()
