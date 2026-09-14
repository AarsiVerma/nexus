# Nexus — Real-Time Voice-Driven Visual Assistant

Nexus is a real-time AI-powered visual assistant designed to help visually impaired users understand their surroundings through **voice interaction, computer vision, object detection, OCR, currency recognition, and visual question answering**.

The system combines multiple AI technologies into a single hands-free assistant that uses a webcam as its visual input and provides information through spoken responses.

---

## Features

- **Wake-word activation** using Vosk
- **Voice question recognition** using OpenAI Whisper
- **Real-time object detection** using YOLOv8n
- **Object counting**
- **Printed text recognition** using EasyOCR
- **Indian currency recognition**
- **Visual Question Answering (VQA)** using Ollama + Moondream V2
- **Voice responses**
  - Windows: SAPI5 through `pyttsx3`
  - macOS: built-in `say`
- **Real-time webcam processing**
- **Shared microphone architecture** for reliable repeated interactions
- **Cross-platform support** for Windows and macOS

---

## Problem Statement

Visually impaired individuals often face difficulties in identifying objects, reading printed text, recognizing currency, and understanding their immediate surroundings independently.

Many existing accessibility solutions may require specialized hardware or provide limited interaction capabilities.

There is a need for an accessible, hands-free system that can use commonly available hardware such as a webcam, microphone, and computer to provide real-time information about the user's surroundings.

---

## Proposed Solution

Nexus is a voice-driven visual assistance system that combines **computer vision, speech recognition, optical character recognition, and multimodal AI**.

The user activates Nexus using the wake phrase:

> **"Hey Nexus"**

After activation, the system listens to the user's question, converts the speech into text, determines the appropriate processing method, analyzes the current camera frame, and provides the result through voice output.

The system is designed as a modular AI pipeline where different technologies handle different tasks.

---

## System Architecture

```text
                         ┌──────────────────┐
                         │   Webcam Input   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     YOLOv8n      │
                         │ Object Detection │
                         └────────┬─────────┘
                                  │
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
      ┌──────────────────┐                   ┌──────────────────┐
      │   Voice Input    │                   │  Current Frame   │
      │    Microphone    │                   │                  │
      └────────┬─────────┘                   └────────┬─────────┘
               │                                      │
               ▼                                      │
      ┌──────────────────┐                            │
      │ Vosk Wake Word   │                            │
      │   "Hey Nexus"    │                            │
      └────────┬─────────┘                            │
               │                                      │
               ▼                                      │
      ┌──────────────────┐                            │
      │     Whisper      │                            │
      │  Speech-to-Text  │                            │
      └────────┬─────────┘                            │
               │                                      │
               ▼                                      ▼
      ┌────────────────────────────────────────────────────────┐
      │                    Question Router                     │
      └──────────────┬──────────────┬──────────────┬───────────┘
                     │              │              │
                     ▼              ▼              ▼
                  OCR /         YOLO Count      Moondream
                Currency          Objects        Visual Q&A
                     │              │              │
                     └──────────────┴──────────────┘
                                    │
                                    ▼
                            ┌─────────────────┐
                            │  Voice Response │
                            │      TTS        │
                            └─────────────────┘
```

---

## Technologies Used

### Programming Language

- **Python** — Core programming language used to develop and integrate the application.

### Artificial Intelligence & Machine Learning

- **YOLOv8n (Ultralytics)** — Real-time object detection and object counting.
- **Moondream V2** — Vision-language model used for image understanding and Visual Question Answering.
- **OpenAI Whisper** — Speech-to-text conversion for understanding user voice commands.
- **Vosk** — Lightweight speech recognition used for wake-word detection.
- **PyTorch** — Deep learning framework used for model execution.

### Computer Vision

- **OpenCV** — Webcam access, real-time video capture, frame processing, and image handling.
- **NumPy** — Numerical and image-array processing.

### Optical Character Recognition

- **EasyOCR** — Extracts and recognizes printed text from camera frames.
- Used for text reading and assisting with Indian currency recognition.

### Voice & Audio Processing

- **SoundDevice** — Captures microphone input and manages audio streams.
- **pyttsx3** — Text-to-speech conversion.
- **Windows SAPI5** — Voice synthesis on Windows.
- **macOS `say`** — Built-in speech synthesis on macOS.

### AI Model Runtime

- **Ollama** — Local runtime used to run the Moondream V2 vision-language model.

### Development Tools

- **Git** — Version control.
- **GitHub** — Source code hosting and collaboration.
- **Python Virtual Environment (`venv`)** — Dependency isolation.
- **PowerShell / Command Prompt** — Windows development and execution.
- **Terminal / Bash** — macOS development and execution.

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core application development |
| Object Detection | YOLOv8n | Real-time object detection |
| Object Counting | YOLOv8n | Counting detected objects |
| Speech Recognition | OpenAI Whisper | Speech-to-text conversion |
| Wake Word Detection | Vosk | Detects "Hey Nexus" |
| OCR | EasyOCR | Printed text recognition |
| Currency Recognition | EasyOCR + Validation | Indian currency identification |
| Visual Question Answering | Ollama + Moondream V2 | Image understanding |
| Computer Vision | OpenCV | Webcam and image processing |
| Audio Input | SoundDevice | Microphone input |
| Text-to-Speech | pyttsx3 + SAPI5 | Windows voice responses |
| Text-to-Speech | macOS `say` | macOS voice responses |
| Deep Learning | PyTorch | AI model execution |
| Numerical Processing | NumPy | Image and numerical processing |
| Model Runtime | Ollama | Local Moondream execution |
| Version Control | Git | Source code management |
| Repository | GitHub | Project hosting |

---

## Requirements

### Hardware Requirements

- Computer or laptop
- Webcam or built-in camera
- Microphone
- Speakers or headphones
- Internet connection for initial model and package downloads

> A dedicated GPU is **not required**. Nexus can run using CPU, although AI model inference may be slower.

### Software Requirements

#### Windows

- Windows 10 or Windows 11
- Python 3.10+
- Ollama
- Git (recommended)

#### macOS

- macOS
- Python 3.10+
- Homebrew
- Ollama
- Git (recommended)

> Linux is currently untested and is not officially supported by this project.

---

# Installation

## 1. Clone the Repository

Open PowerShell, Command Prompt, Terminal, or another shell and run:

```bash
git clone https://github.com/AarsiVerma/nexus.git
cd nexus
```

---

# Windows Setup

## 2. Install Python

Install **Python 3.10 or newer**.

Verify the installation:

```powershell
python --version
```

You should see output similar to:

```text
Python 3.x.x
```

---

## 3. Create a Virtual Environment

From the Nexus project directory:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of the execution policy, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 4. Install Ollama

Install **Ollama for Windows**.

Verify the installation:

```powershell
ollama --version
```

---

## 5. Download the Moondream Vision Model

Nexus uses **Moondream V2** through Ollama.

Run:

```powershell
ollama pull moondream:v2
```

Verify that the model is available:

```powershell
ollama list
```

You should see:

```text
moondream:v2
```

---

## 6. Install Python Dependencies

From the Nexus project directory:

```powershell
pip install -r requirements.txt
```

---

## 7. Download the Vosk Wake-Word Model

Download and extract the following Vosk model:

```text
vosk-model-small-en-us
```

Place the extracted folder inside the Nexus project directory.

The project structure should look like:

```text
nexus/
├── vosk-model-small-en-us/
├── nexus.py
├── requirements.txt
└── ...
```

The folder name must be exactly:

```text
vosk-model-small-en-us
```

---

## 8. Run Nexus on Windows

Nexus can be started directly using Python:

```powershell
python nexus.py
```

Or using the provided Windows launcher:

```powershell
.\run.bat
```

When the application starts successfully, you should see:

```text
Nexus is running... Press Q to quit.
```

Nexus will also announce:

```text
Nexus is ready
```

---

# macOS Setup

## 2. Install Python

Verify that Python is installed:

```bash
python3 --version
```

Python 3.10 or newer is recommended.

---

## 3. Create a Virtual Environment

From the Nexus project directory:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 4. Install Ollama

Using Homebrew:

```bash
brew install ollama
```

Start the Ollama service:

```bash
brew services start ollama
```

Verify the installation:

```bash
ollama --version
```

---

## 5. Download the Moondream Vision Model

```bash
ollama pull moondream:v2
```

Verify:

```bash
ollama list
```

You should see:

```text
moondream:v2
```

---

## 6. Install Python Dependencies

From the Nexus project directory:

```bash
pip3 install -r requirements.txt
```

---

## 7. Download the Vosk Wake-Word Model

Download and extract:

```text
vosk-model-small-en-us
```

Place it inside the Nexus project root:

```text
nexus/
├── vosk-model-small-en-us/
├── nexus.py
├── requirements.txt
└── ...
```

---

## 8. Run Nexus on macOS

Make the launcher executable:

```bash
chmod +x run.sh
```

Run Nexus:

```bash
./run.sh
```

Alternatively, run the application directly:

```bash
python3 nexus.py
```

---

# How to Use

Once Nexus is running, the system continuously processes the webcam feed and listens for the wake phrase.

## Step 1 — Activate Nexus

Say:

```text
Hey Nexus
```

Nexus responds:

```text
I'm listening
```

---

## Step 2 — Ask a Question

After Nexus responds, ask a question about the current camera view.

For example:

```text
What is this?
```

Nexus captures the current camera frame and sends the request to the appropriate processing component.

---

# Example Commands

## General Visual Question Answering

Say:

```text
Hey Nexus
```

Then:

```text
What is this?
```

You can also ask:

```text
What do you see?
```

or:

```text
What is the person holding?
```

Moondream analyzes the current camera frame and generates a natural-language response.

---

## Object Counting

Say:

```text
Hey Nexus
```

Then:

```text
How many bottles are there?
```

Nexus uses YOLO detections from the current scene to count supported objects.

---

## Reading Text

Say:

```text
Hey Nexus
```

Then:

```text
Read this.
```

EasyOCR extracts visible printed text from the camera frame and Nexus reads the result aloud.

---

## Currency Recognition

Show an Indian currency note to the camera and say:

```text
Hey Nexus
```

Then:

```text
How much is this?
```

Nexus uses OCR and currency-specific validation to identify the denomination.

---

# Main Components

## YOLOv8n

YOLOv8n performs real-time object detection using the webcam feed.

It is also used for object counting.

The detection system maintains recent detections so that object-related questions can be answered without running a separate model inference for every voice question.

---

## Vosk

Vosk continuously listens for the wake phrase:

```text
Hey Nexus
```

The application uses a shared microphone stream so wake-word detection and question recording can operate without repeatedly opening separate audio streams.

---

## Whisper

After the wake word is detected, Nexus records the user's question and sends the audio to Whisper for speech-to-text conversion.

Example:

```text
User:
"What is this?"

        ↓

Whisper:

"What is this?"
```

The resulting text is then passed to the question router.

---

## EasyOCR

EasyOCR is used for:

- Printed text recognition
- Currency-related text
- Denomination recognition

The application combines OCR signals with currency-specific validation when identifying Indian currency.

---

## Ollama + Moondream

For general visual questions, Nexus saves the current camera frame and sends it to:

```text
moondream:v2
```

through Ollama.

Example:

```text
"What is the person holding?"
```

Moondream analyzes the image and generates a natural-language answer.

---

## Text-to-Speech

### Windows

Nexus uses:

```text
pyttsx3 → Windows SAPI5
```

A fresh SAPI5 engine is initialized for each spoken response to improve reliability during repeated interactions.

### macOS

Nexus uses the built-in macOS:

```text
say
```

command.

---

# Runtime Flow

A typical Nexus interaction follows this sequence:

```text
1. Webcam captures the environment
                 ↓
2. YOLO continuously detects objects
                 ↓
3. Vosk listens for "Hey Nexus"
                 ↓
4. User says "Hey Nexus"
                 ↓
5. Nexus responds "I'm listening"
                 ↓
6. User asks a question
                 ↓
7. Whisper converts speech → text
                 ↓
8. Question Router determines the required system
                 ↓
          ┌──────┼─────────┬───────────┐
          ↓      ↓         ↓           ↓
         OCR  Currency    YOLO      Moondream
          │      │         │           │
          └──────┴─────────┴───────────┘
                         ↓
                  Generated Answer
                         ↓
                         TTS
                         ↓
                  Spoken Response
```

---

# Project Structure

```text
nexus/
│
├── nexus.py
│   └── Main Nexus application
│
├── wake_word.py
│   └── Standalone wake-word testing
│
├── question_recorder.py
│   └── Standalone audio recording/testing
│
├── vqa_test.py
│   └── Historical BLIP-VQA experiment
│
├── voice_vqa_test.py
│   └── Historical voice + BLIP-VQA experiment
│
├── eval_test.py
│   └── Evaluation/testing harness
│
├── diagnose_ollama_moondream.py
│   └── Ollama + Moondream diagnostic test
│
├── requirements.txt
│   └── Python dependencies
│
├── run.sh
│   └── macOS launcher
│
├── run.bat
│   └── Windows launcher
│
├── PROJECT_STATUS.md
│   └── Detailed project status and development notes
│
├── .gitignore
│
└── README.md
```

---

# Troubleshooting

## Ollama Is Not Responding

Check that Ollama is installed:

```bash
ollama --version
```

Check the available models:

```bash
ollama list
```

Make sure the following model is available:

```text
moondream:v2
```

If it is missing, download it:

```bash
ollama pull moondream:v2
```

---

## Camera Is Not Detected

Make sure:

- Your webcam is connected.
- No other application is using the webcam.
- Camera permissions are enabled.
- The correct camera is selected.

Nexus currently uses the default camera:

```python
cv2.VideoCapture(0)
```

---

## Microphone Is Not Working

Make sure:

- The microphone is connected.
- Your operating system has granted microphone access to Python.
- No other application is exclusively using the microphone.

The project also includes standalone audio recording/testing utilities.

---

## Windows TTS Is Not Working

Nexus uses:

```text
pyttsx3 + Windows SAPI5
```

Make sure Windows has at least one installed speech voice.

Check:

```text
Windows Settings
→ Accessibility
→ Speech
```

After changing speech settings, restart Nexus.

---

## macOS TTS Is Not Working

Test the built-in macOS speech system:

```bash
say "Hello, this is Nexus"
```

If this does not produce audio, the issue is related to the macOS speech or audio configuration rather than Nexus.

---

## CPU Warning

You may see a message such as:

```text
Neither CUDA nor MPS are available - defaulting to CPU.
```

This is **not an error**.

Nexus can run using CPU, although AI model inference may be slower.

---

# Known Limitations

- Real-time performance depends on CPU/GPU hardware.
- YOLO only detects objects supported by its trained classes.
- Moondream's visual answers may occasionally be inaccurate.
- Whisper may occasionally misinterpret speech, especially in noisy environments.
- The current system primarily expects English voice commands.
- Linux has not been officially tested.
- The system currently depends on a webcam and local computer hardware.
- Exact real-world object distance and speed are not currently measured.
- Nexus is a research/prototype project and should not be treated as a certified mobility or safety device.

---

# Future Improvements

The following features are planned for future versions of Nexus.

## Personalized Object Learning

Allow users to confirm objects detected by Nexus and add them to a personalized dataset.

Example workflow:

```text
Nexus detects an object
          ↓
"Is this a water bottle?"
          ↓
User confirms
          ↓
Save image + object label
          ↓
Personalized Dataset
          ↓
Future Custom Model Training
```

---

## User-Confirmed Object Dataset

Allow users to build a personalized dataset by confirming objects that are important to them.

This could help Nexus recognize personal belongings and frequently used objects.

---

## Custom YOLO Training

Use user-confirmed images to train a personalized YOLO model.

Potential custom classes could include:

- Personal belongings
- Household objects
- Frequently used items
- Accessibility-related objects

---

## Real-Time Object Tracking

Add object tracking to monitor movement across consecutive frames.

This can help Nexus identify whether an object is moving toward or away from the user.

---

## Improved Obstacle Detection

Improve the system's ability to detect obstacles in the user's path and provide short voice warnings.

Example:

```text
"Obstacle ahead."
```

or:

```text
"Vehicle approaching from the right."
```

---

## Directional Navigation Assistance

Divide the camera view into three regions:

```text
┌──────────┬──────────┬──────────┐
│   LEFT   │  CENTER  │  RIGHT   │
└──────────┴──────────┴──────────┘
```

Nexus could use this information to provide simple directional guidance.

Example:

```text
"Obstacle ahead. Move slightly left."
```

---

## Approaching-Object Warnings

Combine object detection and tracking to identify objects that appear to be approaching the user.

Potential warnings include:

```text
"Person approaching from the left."
```

or:

```text
"Vehicle approaching from the right."
```

---

## Mobile Integration

Future versions could connect Nexus to a smartphone to provide:

- Mobile camera input
- Portable voice interaction
- Remote processing
- GPS/location integration
- Mobile notifications

---

## Multilingual Voice Interaction

Expand the system to support multiple languages for speech recognition and voice responses.

---

## More Robust Offline Operation

Increase the amount of processing that can be performed locally without requiring external services.

---

## Personalized Scene Understanding

Develop personalized scene understanding based on the user's environment and frequently encountered objects.

---

# Development Notes

The project previously experimented with **BLIP-VQA**.

The current production approach uses:

```text
Ollama + Moondream V2
```

The BLIP-related test files remain in the repository for historical comparison and experimentation, but they are not part of the production Nexus pipeline.

The main production application is:

```text
nexus.py
```

---

# Safety Notice

Nexus is an experimental accessibility prototype.

Its computer-vision and AI-generated predictions can be incorrect, delayed, or incomplete.

Nexus should **not** be relied upon as the sole source of information when:

- Crossing roads
- Navigating traffic
- Using stairs
- Moving through hazardous environments
- Handling emergencies
- Making safety-critical decisions

Users should always use appropriate safety measures and independent judgment.

---

# Contributors

This project was developed as a group project.

Contributions include:

- Computer vision
- Machine learning
- Speech processing
- OCR
- AI model integration
- System development
- Testing
- Documentation

---

# License

This project is intended for **educational and research purposes**.

If this project is later released under a specific open-source license, the appropriate license should be added here.

---

# Acknowledgements

Nexus uses and builds upon the following open-source technologies and models:

- OpenCV
- Ultralytics YOLO
- OpenAI Whisper
- Vosk
- EasyOCR
- Ollama
- Moondream
- PyTorch
- NumPy
- pyttsx3
- SoundDevice

---

# Project Status

Nexus is currently an **active prototype under development**.

The current version provides a functional voice-driven visual assistance pipeline with:

- Wake-word detection
- Speech recognition
- Real-time object detection
- Object counting
- OCR
- Indian currency recognition
- Visual Question Answering
- Voice responses
- Windows and macOS support

Further development will focus on personalized object learning, improved obstacle awareness, object tracking, directional assistance, and mobile integration.
