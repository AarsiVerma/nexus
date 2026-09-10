import numpy as np
import sounddevice as sd
import whisper
import time

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500     # RMS amplitude below this counts as silence
SILENCE_DURATION = 1.5      # stop after this many seconds of silence
MAX_DURATION = 15           # hard cap so it never records forever
CHUNK_DURATION = 0.25       # seconds per audio chunk read from the mic


def record_question():
    print("Recording... speak your question.")
    chunks = []
    silence_start = None
    start_time = time.time()

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    stream.start()

    while True:
        data, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
        chunks.append(data.copy())

        rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_DURATION:
                break
        else:
            silence_start = None

        if time.time() - start_time > MAX_DURATION:
            break

    stream.stop()
    stream.close()

    audio = np.concatenate(chunks, axis=0).flatten()
    # Whisper expects float32 samples in [-1, 1]
    audio = audio.astype(np.float32) / 32768.0
    return audio


def main():
    print("Loading Whisper 'base' model...")
    model = whisper.load_model("base")
    print("Ready.")

    while True:
        input("Press Enter to record a question (Ctrl+C to quit)... ")
        audio = record_question()
        print("Transcribing...")
        result = model.transcribe(audio, language="en", fp16=False)
        print(f"You said: {result['text'].strip()}")


if __name__ == "__main__":
    main()
