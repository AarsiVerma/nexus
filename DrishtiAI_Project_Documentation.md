# DrishtiAI — Real-Time Visual Assistant for the Visually Impaired

Full Project Documentation | ML Research Project

> **Historical document — not the current project state.** This is the
> original project proposal, pasted at the very start of development under
> the name "DrishtiAI." The project was later renamed **Nexus** (via an
> intermediate name, "VisionSaathi") and the actual implementation diverged
> from this plan in several real ways -- most notably, the vision-language
> model used is **moondream2 (via Ollama)**, not BLIP-2/LLaVA, and OCR-based
> text/currency reading and YOLO-based counting were added and were not
> part of this original plan. **For the current, accurate project state,
> see `PROJECT_STATUS.md` and `README.md` instead of this file.** This
> document is kept for reference as the original research proposal, not as
> a description of what was actually built.

## Project Overview

- **Project Name:** DrishtiAI
- **Domain:** Computer Vision, Natural Language Processing, Assistive AI
- **Type:** Real-time multimodal AI system
- **Platform:** MacBook (demo) + Google Colab (training)
- **Target Users:** Visually impaired individuals, particularly in Indian environments

DrishtiAI is a voice-driven, camera-based AI assistant that runs continuously in the background. It proactively alerts users to obstacles and hazards detected by the camera, and answers spoken questions about whatever the camera currently sees — including reading medicine bottles, identifying currency notes, describing surroundings, and reading Hindi/English signboards.

## Problem Statement

Over 8 million Indians are visually impaired, yet existing AI assistive tools such as Microsoft Seeing AI and Google Lookout are designed and tested for Western environments. They fail on:

- Hindi and regional language text on signboards and packaging
- Indian currency notes (₹10 to ₹2000)
- Indian street environments (autorickshaws, two-wheelers, street vendors, narrow lanes)
- Medicine bottles with mixed Hindi-English labels
- Indian food packaging and products

Additionally, all existing tools require the user to capture a photo — an unnatural and impractical interaction for a visually impaired person navigating daily life. A blind person cannot stop, frame a photo, and wait for a response while crossing a road or navigating a market.

DrishtiAI solves both problems: it handles Indian-specific visual content and operates entirely in real-time without requiring any screen interaction.

## Proposed Solution

DrishtiAI is a real-time, hands-free, voice-activated visual assistant. It:

- Runs the camera continuously and automatically detects obstacles, people, vehicles, and hazards without the user asking
- Proactively speaks alerts like "Person ahead" or "Stairs detected"
- Listens for a custom wake word ("Hey Drishti") at all times
- Once activated by voice, records the user's spoken question
- Analyzes the current camera frame along with the question using a Vision-Language Model
- Speaks the answer back aloud immediately

The entire interaction is hands-free and eyes-free. The user never touches a screen.

## System Architecture

The system has two parallel layers running simultaneously:

**Layer 1 — Continuous Detection (always on):**
Camera Feed → Frame Sampling (1 frame/sec) → YOLOv8 Object Detection → Obstacle Alert via TTS

**Layer 2 — On-Demand Q&A (activates on wake word):**
Wake Word Detection → "Hey Drishti" heard → Whisper records question → Frame captured → VLM processes frame + question → Answer spoken aloud

Both layers run in parallel threads. The system is always watching and always listening.

## Complete Tech Stack

| Component | Tool/Library | Purpose |
|---|---|---|
| Camera Feed | OpenCV | Continuous live webcam frame capture |
| Fast Detection | YOLOv8 (Ultralytics) | Real-time obstacle and object detection |
| Wake Word | Porcupine (Picovoice) | "Hey Drishti" custom wake word, runs on-device |
| Speech Input | OpenAI Whisper (local) | Converts user's spoken question to text |
| Vision-Language Model | BLIP-2 or LLaVA | Answers questions about the camera frame |
| Text to Speech | pyttsx3 / macOS say | Speaks answers and alerts aloud |
| Model Fine-Tuning | Google Colab (T4 GPU) | Fine-tune VLM on VizWiz dataset |
| Training Dataset | VizWiz (31,000 images) | Images taken by blind users with QA pairs |
| Indian Evaluation | IDD Dataset, IIIT-ILST | Indian street scenes and text benchmarks |
| Language | Python 3.12 | All components written in Python |

All tools are free and open-source. No paid APIs required.

## Key Features and Capabilities

What the system can do:

- Read medicine bottle labels, dosage instructions, and expiry dates
- Identify Indian currency notes (₹10, ₹50, ₹100, ₹200, ₹500, ₹2000)
- Read Hindi and English text on signboards, menus, product labels
- Identify objects, food items, clothing, and household items
- Detect and warn about obstacles — stairs, steps, walls, open doors
- Warn about nearby vehicles — cars, motorcycles, bicycles
- Describe the overall scene when asked "What is around me?"
- Identify colors — "What color is this shirt?"
- Count people — "How many people are in this room?"
- Activate entirely by voice — no screen interaction ever required

**Wake word feature:** The app runs silently in the background. User says "Hey Drishti" and the system activates. No buttons, no app icon to find, no screen to navigate. This is the core accessibility design principle of the project.

## Agentic AI Framing

DrishtiAI is a multimodal assistive agent. In AI terminology, an "agent" is a system that:

- Perceives its environment (camera + microphone)
- Makes decisions autonomously (when to alert, when to listen)
- Takes actions (speaks alerts, answers questions)
- Operates in a continuous loop without human instruction

DrishtiAI fits this definition precisely — it continuously perceives, decides, and acts without any manual input. This framing should be used in the research paper: "a multimodal real-time assistive agent for visually impaired users."

## Development Plan — Phase by Phase

### Phase 1: Environment Setup (Day 1)

**Goal:** Get Python environment ready with all libraries installed.

Steps:

1. Open Terminal on MacBook
2. Run: `python3 -m venv venv`
3. Run: `source venv/bin/activate`
4. Run: `pip install ultralytics opencv-python pyttsx3 pyobjc-framework-AVFoundation`
5. Verify: `python3 -c "import cv2; import ultralytics; print('Ready')"`

**Done when:** "Ready" prints with no errors.

### Phase 2: Component 1 — Live Camera + Object Detection (Days 2–4)

**Goal:** Webcam runs continuously, YOLO detects objects, system speaks alerts automatically.

**Libraries:** `ultralytics`, `opencv-python`, `pyttsx3`

What to build:

- Open webcam using OpenCV
- Sample one frame per second
- Run YOLOv8 on each frame
- Filter detections by confidence > 50%
- Add 5-second cooldown so same object is not announced repeatedly
- Draw bounding boxes on the display window
- Speak alert when relevant object detected: "Person ahead", "Car nearby", "Stairs detected"

**Test by:** Placing a water bottle, phone, and yourself in front of the camera. System should announce each without you doing anything.

**Done when:** Camera window opens, objects are labeled on screen, and audio alerts fire automatically.

### Phase 3: Component 2 — Wake Word Detection (Days 5–7)

**Goal:** System listens for "Hey Drishti" and activates on hearing it.

**Libraries:** `pvporcupine`, `pyaudio`

Steps:

1. Sign up at console.picovoice.ai (free account)
2. Navigate to Porcupine Wake Word section
3. Type "Hey Drishti" as the wake word, download the `.ppn` model file
4. Get your free API access key from the console
5. Build a standalone script that prints "ACTIVATED" when wake word is heard
6. Test it thoroughly with different volumes and distances
7. Integrate with Component 1: system runs YOLO normally, switches to listening mode when wake word heard

System says: "I'm listening" when activated

**Done when:** You say "Hey Drishti" and the system responds audibly without any touch input.

### Phase 4: Component 3 — Voice Question Recording (Days 8–10)

**Goal:** After wake word, system records user's question and converts to text.

**Libraries:** `openai-whisper`, `sounddevice`, `scipy`

Install: `pip install openai-whisper sounddevice scipy`

What to build:

- After wake word activates, start recording audio
- Stop recording when 1.5 seconds of silence detected
- Pass audio to Whisper "base" model for transcription
- Print transcribed text to Terminal to verify accuracy
- Test with Indian-accented questions in English: "What is in front of me?", "What does this label say?", "How much is this note?"

**Done when:** You say "Hey Drishti" then ask "What is on the table?" and the correct text appears in Terminal.

### Phase 5: Component 4 — Vision Language Model (Days 11–16)

**Goal:** The brain of the system. Looks at current camera frame + transcribed question, generates a natural language answer.

**Libraries:** `transformers`, `torch`, `Pillow`, `accelerate`

Install: `pip install transformers torch torchvision Pillow accelerate`

What to build:

- Load BLIP-2 model from HuggingFace (downloads ~3GB on first run)
- Pass a test image + a test question, print the answer
- Test on Indian objects: medicine bottle, ₹500 note, Hindi signboard, food packet
- Document where it gets it wrong — these failures are important for the paper
- Connect to live camera: instead of saved image, use current webcam frame
- Connect to Whisper output: transcribed question feeds directly into BLIP-2

**Note on speed:** First answer takes 15–30 seconds while model loads. Subsequent answers take 2–5 seconds. This is acceptable for the demo.

**Done when:** You hold up a medicine bottle, system speaks the dosage correctly.

### Phase 6: Full Integration (Days 17–20)

**Goal:** All 5 components working together as one complete system.

What to build:

- Merge all component scripts into one final file: `drishti_main.py`
- Use Python threading: YOLO detection, wake word listening, and TTS run simultaneously
- Full loop: camera running → YOLO alerts firing → wake word heard → Whisper records question → BLIP-2 answers → TTS speaks → back to monitoring
- Add clean startup: system says "DrishtiAI is ready" on launch
- Test the complete loop multiple times without crashes
- Record a 2-minute demo video of the full system working

**Done when:** Complete hands-free loop runs without manual intervention for 5+ minutes.

### Phase 7: Fine-Tuning on Google Colab (Week 5–6)

**Goal:** Improve the VLM by fine-tuning on VizWiz — images taken by blind users.

This is the core ML research contribution.

Steps:

1. Download VizWiz dataset from vizwiz.org (free, requires email registration)
2. Open Google Colab, select T4 GPU runtime (free)
3. Upload dataset to Google Drive, mount in Colab
4. Load base BLIP-2 from HuggingFace
5. Run baseline evaluation on VizWiz test set — record accuracy score (this is your "before" number)
6. Fine-tune on VizWiz training set for 3–5 epochs
7. Run evaluation again on same test set — record new accuracy (this is your "after" number)
8. Save fine-tuned model weights to Google Drive
9. Download weights, load into main demo script

**Key metric:** VQA Accuracy — percentage of questions answered correctly. Even a 5% improvement is a publishable result.

**Training time:** Approximately 3–4 hours on Colab free tier T4 GPU.

### Phase 8: Indian Environment Evaluation (Week 7)

**Goal:** Test how well the model handles Indian-specific scenarios. This is your novel research contribution.

What to do:

- Photograph 30–50 Indian-context images yourself: medicine bottles, ₹ currency notes, Hindi shop signs, Indian food packaging, street scenes
- Write 3–5 questions per image: "What does this say?", "How much is this note?", "What medicine is this?", "What is the price?"
- Run base BLIP-2 on all images, record answers, mark correct/incorrect
- Run fine-tuned BLIP-2 on same images, record answers
- Compare both — create a results table
- Categorize failure types: Hindi text failures, currency errors, scene misidentification
- Also test using IDD dataset images for Indian road scene evaluation

**Output:** A results table showing base model vs fine-tuned model performance on Indian scenarios — the key table in your paper.

### Phase 9: Research Paper Writing (Weeks 8–9)

Write one section per week. Content for each section comes directly from the work already done.

**Paper Structure:**

1. **Abstract** — 150 words, written last. Problem, approach, key results.
2. **Introduction** — Why visually impaired Indians are underserved by existing tools. What DrishtiAI does. Paper contributions listed clearly.
3. **Related Work** — Cite: VizWiz challenge papers, BLIP-2 paper, LLaVA paper, Microsoft Seeing AI, Lookout by Google, other VQA papers. Minimum 8 citations.
4. **System Architecture** — Describe the 5-component pipeline. Include a flow diagram. Explain the two-layer design (YOLO + VLM). Explain why wake word is essential for accessibility.
5. **Methodology** — How you fine-tuned BLIP-2. Dataset details, training setup, hyperparameters, evaluation metrics (VQA Accuracy, WUPS Score).
6. **Experiments and Results** — Table: Base model vs Fine-tuned model on VizWiz test set. Table: Both models on Indian evaluation set. Latency measurements.
7. **Indian Context Analysis** — Where models succeed and fail on Indian inputs. Failure categories with image examples. Discussion of why these failures occur.
8. **Limitations** — Model latency on MacBook, Hindi language gaps, mobile deployment not implemented, Indian dataset size is small.
9. **Future Work** — Fine-tune specifically on Indian visual data, deploy on Android, add Hindi voice input/output, optimize for edge devices.
10. **Conclusion** — Summary of contribution: real-time system + fine-tuning improvement + Indian evaluation benchmark.

### Phase 10: Demo Preparation (Week 10)

**Demo script for presentation:**

1. Show the system launching — says "DrishtiAI is ready"
2. Place a water bottle in front of camera — system announces "Bottle nearby" automatically
3. Walk toward the camera — system announces "Person detected ahead"
4. Say "Hey Drishti" — system says "I'm listening"
5. Ask: "What is in front of me?" — system describes the scene
6. Hold up a medicine bottle — ask "What is the dosage?" — system reads it aloud
7. Hold up a ₹500 note — ask "How much is this?" — system answers "Five hundred rupees"
8. Hold up a Hindi sign — ask "What does this say?" — system reads it
9. Close the laptop — demonstrate it works entirely by voice

**Demo duration:** 3–5 minutes is enough to show all features.

## Full Timeline

| Week | Phase | Goal |
|---|---|---|
| Week 1 | Environment + Component 1 | Camera running, YOLO detecting and alerting |
| Week 2 | Components 2 and 3 | Wake word + Whisper working |
| Week 3 | Component 4 | BLIP-2 answering questions from camera |
| Week 4 | Full Integration | Complete hands-free loop working |
| Week 5–6 | Colab Fine-Tuning | Fine-tuned model, before/after results |
| Week 7 | Indian Evaluation | Results table for Indian scenarios |
| Week 8–9 | Paper Writing | Full paper drafted |
| Week 10 | Demo + Submission | Final polish and submission |

## Datasets Used

| Dataset | What it contains | Where to get it |
|---|---|---|
| VizWiz | 31,000 images taken by blind users with QA pairs | vizwiz.org (free, email signup) |
| IDD | Indian road and street scene images | idd.insaan.iiit.ac.in (free) |
| IIIT-ILST | Indian language text in scene images | cvit.iiit.ac.in (free) |
| Custom (self-collected) | 30–50 Indian objects photographed by you | Photograph yourself in 1 hour |

## Research Paper Title

"DrishtiAI: A Real-Time Multimodal Assistive Agent for Visually Impaired Users in Indian Environments Using Fine-Tuned Vision-Language Models"

## How to Frame the ML Contribution

The ML contribution of this project has three parts:

1. Fine-tuning BLIP-2 on VizWiz and measuring the improvement in VQA accuracy (quantitative result)
2. Evaluating both base and fine-tuned models on Indian-specific visual scenarios and documenting performance gaps (novel evaluation)
3. Proposing and implementing a two-tier real-time agent architecture (YOLO + VLM) for practical assistive deployment (system contribution)

No other published paper has evaluated VQA models specifically on Indian assistive use cases. That is the originality claim.
