import cv2
import time
import ollama

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

cv2.imwrite("/tmp/ollama_test_frame.jpg", frame)
print("Frame captured, saved to /tmp/ollama_test_frame.jpg")

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
            "images": ["/tmp/ollama_test_frame.jpg"],
        }],
    )
    elapsed = time.time() - start
    answer = response["message"]["content"]
    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print(f"(took {elapsed:.1f}s)")
