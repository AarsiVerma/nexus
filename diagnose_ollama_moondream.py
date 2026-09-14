import cv2
import time
import ollama
import os
import tempfile

print("Capturing one frame from webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
for _ in range(5):
    ret, frame = cap.read()
cap.release()

if not ret:
    print("FAILED to capture a frame from the webcam.")
    raise SystemExit(1)

frame_path = os.path.join(tempfile.gettempdir(), "ollama_test_frame.jpg")
cv2.imwrite(frame_path, frame)
print(f"Frame captured, saved to {frame_path}")

questions = [
    "Describe everything you see in this image in detail.",
    "What is this?",
    "How many people are there?",
    "What color is the background?",
    "What is around me?",
]

for q in questions:
    start = time.time()
    response = ollama.chat(
        model="moondream:v2",
        messages=[{
            "role": "user",
            "content": q,
            "images": [frame_path],
        }],
    )
    elapsed = time.time() - start
    answer = response["message"]["content"]
    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print(f"(took {elapsed:.1f}s)")
