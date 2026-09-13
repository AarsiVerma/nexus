import sys
import re
import cv2
import ollama
import easyocr
from ultralytics import YOLO

# --- Mirrors vision_saathi.py's routing logic exactly, without importing it
# (that file runs a live camera loop at import time) -- kept in sync by hand
# for this one-off evaluation. -----------------------------------------

VQA_MODEL = "moondream:v2"

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
READ_TRIGGERS = ('read', 'says', 'written', 'say', 'text on', 'label say')
CURRENCY_TRIGGERS = (
    'how much money', 'how much is this', 'what note', 'which note',
    'what currency', 'rupee', 'rupees',
)
INDIAN_DENOMINATIONS = {'10', '20', '50', '100', '200', '500', '2000'}

print("Loading YOLO...")
yolo_model = YOLO('yolov8n.pt')
print("Loading OCR reader...")
ocr_reader = easyocr.Reader(['en', 'hi'])

_question_counter = 0


def make_descriptive_prompt(question):
    global _question_counter
    _question_counter += 1
    q = question.strip()
    if not q.endswith(('.', '?', '!')):
        q += '?'
    if not q.lower().startswith(('can you', 'could you', 'would you', 'please')):
        q = f"Can you tell me {q[0].lower()}{q[1:]}"
    return f"[Question #{_question_counter}] {q}"


def read_text_aloud(frame):
    results = ocr_reader.readtext(frame)
    kept = [(bbox, text) for (bbox, text, confidence) in results if confidence > 0.25]
    if not kept:
        return "I couldn't find any readable text there."
    kept.sort(key=lambda item: (item[0][0][1], item[0][0][0]))
    texts = [text for (_, text) in kept]
    return "It says: " + ", ".join(texts)


def identify_currency(frame):
    results = ocr_reader.readtext(frame)
    kept = [text for (_, text, confidence) in results if confidence > 0.25]
    if not kept:
        return "I couldn't find a currency note clearly in view."
    for text in kept:
        digits = re.sub(r'[^0-9]', '', text)
        if digits in INDIAN_DENOMINATIONS:
            return f"This looks like a {digits} rupee note."
    return "I can see a note but couldn't clearly read the amount. It says: " + ", ".join(kept)


def get_boxes(frame):
    small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
    results = yolo_model(small, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            label = yolo_model.names[int(box.cls)]
            confidence = float(box.conf)
            if confidence > 0.6:
                boxes.append((label, confidence))
    return boxes


def count_known_objects(question, boxes):
    words = re.findall(r"[a-z']+", question.lower())
    for word in words:
        if word in COUNTABLE_OBJECTS:
            label = COUNTABLE_OBJECTS[word]
            count = sum(1 for (box_label, _) in boxes if box_label == label)
            if count == 0:
                return f"I don't see any {word} right now."
            return f"I can see {count} {word} right now."
    return None


def answer_question(question, frame, frame_path):
    q = question.lower()

    if any(trigger in q for trigger in CURRENCY_TRIGGERS):
        return identify_currency(frame)

    if any(trigger in q for trigger in READ_TRIGGERS):
        return read_text_aloud(frame)

    if q.startswith(("how many", "count")):
        boxes = get_boxes(frame)
        count_answer = count_known_objects(q, boxes)
        if count_answer is not None:
            return count_answer

    cv2.imwrite(frame_path, frame)
    response = ollama.chat(
        model=VQA_MODEL,
        messages=[{
            "role": "user",
            "content": make_descriptive_prompt(question),
            "images": [frame_path],
        }],
        options={"num_predict": 200},
    )
    return response["message"]["content"].strip()


# --- Test runner -----------------------------------------------------

if __name__ == "__main__":
    frame_path = sys.argv[1]
    questions = sys.argv[2:]

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Could not read frame at {frame_path}")
        sys.exit(1)

    for q in questions:
        answer = answer_question(q, frame, "/tmp/eval_vqa_frame.jpg")
        print(f"\nQ: {q}")
        print(f"A: {answer}")
