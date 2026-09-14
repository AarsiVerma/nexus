# Nexus — Real-Time Voice-Driven Visual Assistant

Nexus is a real-time AI-powered visual assistant designed to help visually impaired users understand their surroundings through **voice interaction, computer vision, OCR, and visual question answering**.

The system combines object detection, speech recognition, OCR, currency recognition, and multimodal AI into a single hands-free assistant.

## Features

- **Wake-word activation** using Vosk
- **Voice question recognition** using OpenAI Whisper
- **Real-time object detection** using YOLOv8n
- **Object counting**
- **Printed text recognition** using EasyOCR
- **Indian currency recognition**
- **Visual question answering** using Ollama + Moondream
- **Voice responses**
  - Windows: SAPI5 through `pyttsx3`
  - macOS: built-in `say`
- **Real-time webcam processing**
- Shared microphone architecture for reliable repeated interactions
- Cross-platform support for **Windows and macOS**

---

# System Architecture

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
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌──────────────────┐                     ┌──────────────────┐
│  Voice Input     │                     │ Current Frame    │
│  Microphone      │                     │                  │
└────────┬─────────┘                     └────────┬─────────┘
         │                                        │
         ▼                                        │
┌──────────────────┐                              │
│ Vosk Wake Word   │                              │
│  "Hey Nexus"     │                              │
└────────┬─────────┘                              │
         │                                        │
         ▼                                        │
┌──────────────────┐                              │
│     Whisper      │                              │
│ Speech-to-Text   │                              │
└────────┬─────────┘                              │
         │                                        │
         ▼                                        ▼
┌─────────────────────────────────────────────────────────┐
│                    Question Router                      │
└───────────────┬───────────────┬───────────────┬─────────┘
                │               │               │
                ▼               ▼               ▼
             OCR /          YOLO Count       Moondream
           Currency           Objects        Visual Q&A
                │               │               │
                └───────────────┴───────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Voice Response │
                       │      TTS        │
                       └─────────────────┘
Technology Stack
Component	Technology
Object Detection	YOLOv8n
Speech Recognition	OpenAI Whisper
Wake Word	Vosk
OCR	EasyOCR
Visual Question Answering	Ollama + Moondream
Text-to-Speech — Windows	pyttsx3 + SAPI5
Text-to-Speech — macOS	macOS say
Computer Vision	OpenCV
Audio Input	SoundDevice
Language	Python


Requirements
Hardware
- Computer with a webcam
- Microphone
- Speakers or headphones
- Internet connection for initial model/package downloads
A dedicated GPU is not required. The application can run using CPU, although model inference will be slower.
Software
Windows
- Windows 10/11
- Python 3.10+
- Ollama
- Git (recommended)
macOS
- macOS
- Python 3.10+
- Homebrew
- Ollama
- Git (recommended)
Linux is currently untested and is not officially supported by this project.

Installation
1. Clone the Repository
git clone https://github.com/AarsiVerma/nexus.git
cd nexus
Windows Setup
2. Install Python
Install Python 3.10 or newer.
Verify the installation:
python --version
You should see something similar to:
Python 3.x.x
3. Install Ollama
Install Ollama for Windows from the official Ollama website.
After installation, verify:
ollama --version
Pull the Moondream vision model:
ollama pull moondream:v2
Verify that it is available:
ollama list
You should see:
moondream:v2
4. Install Python Dependencies
From the Nexus project directory:
pip install -r requirements.txt
5. Download the Vosk Wake-Word Model
Download:
vosk-model-small-en-us
Extract the model so that the project structure contains:
nexus/
├── vosk-model-small-en-us/
├── nexus.py
├── requirements.txt
└── ...
The folder name must match:
vosk-model-small-en-us
6. Run Nexus on Windows
You can run Nexus directly:
python nexus.py
Or use the Windows launcher:
.\run.bat
When everything is initialized, you should see:
Nexus is running... Press Q to quit.
Nexus will announce that it is ready.
macOS Setup
2. Install Python
Verify Python:
python3 --version
Python 3.10 or newer is recommended.
3. Install Ollama
Install Ollama using Homebrew:
brew install ollama
Start the Ollama service:
brew services start ollama
Pull the Moondream model:
ollama pull moondream:v2
Verify:
ollama list
4. Install Python Dependencies
From the Nexus directory:
pip3 install -r requirements.txt
5. Download the Vosk Wake-Word Model
Download and extract:
vosk-model-small-en-us
Place it in the project root:
nexus/
├── vosk-model-small-en-us/
├── nexus.py
├── requirements.txt
└── ...
6. Run Nexus on macOS
Use the provided launcher:
./run.sh
If necessary:
chmod +x run.sh
Then:
./run.sh
You can also run the application directly with Python if your environment is configured correctly:
python3 nexus.py
Using Nexus
Once Nexus is running:
1. Activate Nexus
Say:
Hey Nexus
Nexus responds:
I'm listening
2. Ask a question
For example:
What is this?
or:
What do you see?
Nexus captures the current camera frame and uses the appropriate vision component to answer.
Example Commands
General Visual Question Answering
Hey Nexus

What is this?
Moondream analyzes the current camera frame and provides a spoken description.
Object Counting
Hey Nexus

How many bottles are there?
Nexus uses the YOLO detections from the current scene to count supported objects.
Reading Text
Hey Nexus

Read this.
EasyOCR extracts visible printed text and reads it aloud.
Currency Recognition
Show an Indian currency note and ask:
Hey Nexus

How much is this?
Nexus uses OCR and currency-specific validation to identify the denomination.
Main Components
YOLOv8n
YOLOv8n performs real-time object detection using the webcam feed.
It is also used for object counting.
The detection system maintains recent detections so that questions about objects can be answered without running a separate model inference for every voice question.
Vosk
Vosk continuously listens for the wake phrase:
Hey Nexus
The application uses a shared microphone stream so wake-word detection and question recording can operate without repeatedly opening separate audio streams.
Whisper
After the wake word is detected, Nexus records the user's question and sends the audio to Whisper for speech-to-text conversion.
Example:
User:
"What is this?"

Whisper:
"What is this?"
The resulting text is passed to the question router.
EasyOCR
EasyOCR is used for:
- Printed text
- Currency-related text
- Denomination recognition
The application combines multiple OCR signals when identifying Indian currency to reduce false positives.
Ollama + Moondream
For general visual questions, Nexus saves the current camera frame and sends it to:
moondream:v2
through Ollama.
Example:
"What is the person holding?"
Moondream analyzes the image and generates a natural-language answer.
Text-to-Speech
Windows
Nexus uses:
pyttsx3 → Windows SAPI5
A fresh SAPI5 engine is initialized for each spoken response to improve reliability during repeated interactions.
macOS
Nexus uses the macOS built-in:
say
command.
Project Structure
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
Runtime Flow
A typical interaction follows this sequence:
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
8. Question router determines the required system
              ↓
      ┌───────┼────────┬──────────┐
      ↓       ↓        ↓          ↓
     OCR    Currency  YOLO     Moondream
      │       │        │          │
      └───────┴────────┴──────────┘
                   ↓
             Generated answer
                   ↓
                 TTS
                   ↓
             Spoken response
Troubleshooting
Ollama is not responding
Check that Ollama is running:
ollama list
or on macOS:
ollama list
Make sure the model exists:
moondream:v2
If it is missing:
ollama pull moondream:v2
Camera is not detected
Make sure:
- Your webcam is connected.
- No other application is using the webcam.
- Camera permissions are enabled for Python/your terminal.
Nexus currently uses the default camera:
cv2.VideoCapture(0)
Microphone is not working
Check that your operating system has granted microphone access to Python/your terminal.
You can also test the microphone using the standalone recording utilities included in the project.
Windows TTS is not working
Nexus uses:
pyttsx3 + Windows SAPI5
Make sure Windows has at least one installed speech voice.
You can check:
Windows Settings
→ Accessibility
→ Speech
Restart Nexus after changing speech settings.
macOS TTS is not working
Test the built-in macOS speech system:
say "Hello, this is Nexus"
If this does not produce audio, the problem is with the macOS speech/audio configuration rather than Nexus.
CPU warning
You may see a message such as:
Neither CUDA nor MPS are available - defaulting to CPU.
This is not an error.
Nexus can run on CPU, although inference may be slower.
Known Limitations
- Real-time performance depends on CPU/GPU hardware.
- YOLO only detects objects supported by its trained classes.
- Moondream's visual answers may occasionally be inaccurate.
- Whisper may occasionally misinterpret speech, especially in noisy environments.
- The system currently expects English voice commands.
- Linux has not been officially tested.
- The current system is a research/prototype project and should not be treated as a certified mobility or safety device.
Future Improvements
Planned improvements include:
- Personalized object learning
- User-confirmed object datasets
- Custom YOLO training
- Real-time object tracking
- Improved obstacle detection
- Directional navigation assistance
- Approaching-object warnings
- Mobile camera integration
- Improved multilingual voice interaction
- More robust offline operation
- Personalized scene understanding
Development Notes
The project previously experimented with BLIP-VQA. That approach was abandoned in favor of:
Ollama + Moondream
The BLIP-related test files remain in the repository for historical comparison and experimentation, but they are not part of the production Nexus pipeline.
The production application is:
nexus.py
Safety Notice
Nexus is an experimental accessibility prototype.
Its computer-vision and navigation-related predictions can be incorrect or delayed. It should not be relied upon as the sole source of information when navigating roads, traffic, stairs, or other hazardous environments.
License
This project is intended for educational and research purposes.

### One thing I deliberately changed

I included this in **Future Improvements**:

> Personalized object learning  
> User-confirmed object datasets  
> Custom YOLO training  
> Real-time object tracking  
> Directional navigation assistance

That's useful because those are exactly the features we're considering next, without pretending they're already implemented.

Also, I would **not claim that Nexus can currently calculate vehicle speed or safely tell a person which way to walk**. Those should remain future features until we actually implement and test them.

Your current checkpoint is therefore nicely positioned: **working cross-platform core first, then new accessibility features on top.**




  






















Thinking effortGPT-5.6 Sol




























    To pick up a draggable item, press the space bar.
    While dragging, use the arrow keys to move the item.
    Press space again to drop the item in its new position, or press escape to cancel.
