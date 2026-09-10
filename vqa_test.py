import cv2
import time
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

print("Loading BLIP VQA model...")
VQA_MODEL_PATH = "blip_finetuned"
processor = BlipProcessor.from_pretrained(VQA_MODEL_PATH)
model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_PATH)
print("Ready.")

cap = cv2.VideoCapture(0)

while True:
    question = input("\nType a question about what the camera sees (or 'q' to quit): ")
    if question.lower() == 'q':
        break

    ret, frame = cap.read()
    if not ret:
        print("Could not read from camera")
        continue

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    start = time.time()
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    answer = processor.decode(out[0], skip_special_tokens=True)
    elapsed = time.time() - start

    print(f"Answer: {answer}  (took {elapsed:.1f}s)")

cap.release()
