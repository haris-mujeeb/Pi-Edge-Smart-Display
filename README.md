# Pi-Edge-Smart-Display

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hardware: Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi-C51A4A?logo=raspberry-pi)](https://www.raspberrypi.com/)

## Description
**Pi-Edge-Smart-Display** is a fully localized, privacy-first smart assistant and interactive display. It leverages the compute power of a Raspberry Pi combined with a Google Coral Edge TPU to deliver real-time voice interaction, edge-based computer vision, and dynamic UI updates—all without relying on cloud processing or external APIs. 

The system orchestrates a localized Large Language Model (LLM) for reasoning, accelerated speech-to-text/text-to-speech pipelines, and hardware-accelerated face detection, ensuring user data never leaves the device.

## Components

### Hardware
* **Compute:** Raspberry Pi (4/5)
* **Vision Accelerator:** Google Coral Edge TPU (USB)
* **Display:** 7-inch HDMI Display
* **Camera:** OV5647 Camera Module (CSI)
* **Audio Input:** INMP441 Omnidirectional Microphone (I2S)
* **Audio Output:** MAX98357A I2S Amplifier + Speaker

### Software Stack
* **Vision / Face Detection:** TensorFlow Lite (Edge TPU Runtime)
* **Wake Word:** openWakeWord / Porcupine
* **Speech-to-Text (ASR):** Whisper.cpp
* **Reasoning (LLM):** Ollama / llama.cpp (e.g., Qwen 1.5B, Llama-3.2-1B)
* **Text-to-Speech (TTS):** Piper TTS
* **Orchestration Backend:** Python + FastAPI
* **Frontend UI:** React / Vue.js (running in Chromium Kiosk Mode)

---

## Project Structure

This repository follows a modular, service-oriented architecture to separate hardware interfaces, AI models, and the web frontend. Below is the directory map showing both currently implemented files and planned/missing files to guide future development.

```text
pi-edge-smart-display/
├── audio_files/            # Directory for recorded and test audio files
│   ├── hello.wav
│   ├── recording.wav
│   └── test_hardware_mic.wav
├── hardware/               # Hardware configuration and physical casing
│   ├── 3d-printed-enclosure/ # STL/3MF files for casing
│   ├── asound.conf         # [Planned] ALSA audio configuration for I2S shared clock
│   └── boot_config.txt     # [Planned] Pi /boot/firmware/config.txt overlays
├── images/                 # Test images and captured camera frames
│   ├── capture.jpg         # Sample captured photo
│   └── test_data/          # Static images for validating face detection
├── models/                 # Local directory for downloaded weights (Git-ignored)
│   ├── face_detection_front.tflite # Face detection model weights
│   ├── llm/                # [Planned] GGUF files for llama.cpp/Ollama
│   ├── vision/             # [Planned] .tflite models compiled for Edge TPU
│   └── voice/              # [Planned] Whisper and Piper model files
├── src/                    # Core Python backend and orchestration
│   ├── api/                # [Planned] FastAPI application and routes
│   ├── audio/              # Audio capture and transcription
│   │   └── audio_to_text_node.py # Whisper-based speech-to-text pipeline
│   ├── llm/                # Local LLM orchestration
│   │   └── llm_node.py     # Prompt queries and Ollama integration
│   ├── vision/             # Edge computer vision
│   │   └── vision_node.py  # Camera feed capture and TF-Lite face detection
│   └── main.py             # Main pipeline orchestrator
├── tests/                  # Backend unit and integration tests (pytest)
│   ├── audio_to_text_node_test.py # Audio transcription tests
│   ├── llm_node_test.py    # Local LLM and mock API tests
│   └── vision_node_test.py # Face detection and camera tests
├── ui/                     # [Planned] Frontend web application
│   ├── public/             # [Planned] Static public assets
│   ├── src/                # [Planned] React/Vue components (Dashboard, Chat, widgets)
│   ├── package.json        # [Planned] Frontend package details
│   └── vite.config.js      # [Planned] Bundler/Vite configuration
├── scripts/                # [Planned] Automation/Utility scripts
│   ├── setup_env.sh        # [Planned] Automated environment installation script
│   └── start_kiosk.sh      # [Planned] Script to launch Chromium in kiosk mode
├── .gitignore
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Project Status & Checklist

Below is the current progress roadmap of the project. This checklist tracks what has been implemented and tested, along with remaining tasks for both hardware and software integrations.

### Completed / Tested Tasks
- [x] **Audio Input Hardware:** Tested recording using the laptop's built-in microphone and the external **INMP441** microphone module via I2S.
- [x] **Speech-to-Text (ASR):** Integrated speech-to-text transcription utilizing `faster_whisper` (specifically the `tiny.en` model).
- [x] **Local LLM Orchestration:** Set up local text generation reasoning using Ollama with the `qwen2.5:1.5b` model.
- [x] **Vision Detection (Laptop):** Verified facial detection using TensorFlow Lite (`face_detection_front.tflite`) on the laptop's built-in webcam.

### Remaining Tasks
- [ ] **Vision Hardware Testing:**
  - [ ] Test face detection on a **Raspberry Pi 4B** using the **OV5647** camera module.
- [ ] **Wake Word Activation:**
  - [ ] Implement and integrate wake word detection (e.g., using openWakeWord or Porcupine).
- [ ] **Text-to-Speech (TTS):**
  - [ ] Implement localized voice generation (e.g., using Piper TTS) to speak response outputs.
- [ ] **Frontend User Interface:**
  - [ ] Build and style the web display dashboard (React/Vue.js running in Chromium Kiosk Mode).
- [ ] **FastAPI Backend & API Server:**
  - [ ] Complete API routes and backend orchestration under `src/api/`.
- [ ] **Hardware Configuration Files:**
  - [ ] Write and deploy ALSA audio configurations (`hardware/asound.conf`) and Raspberry Pi boot overrides (`hardware/boot_config.txt`).

---

## Installation

### Python 3.10 Setup (via uv)

To avoid system package conflicts on your Raspberry Pi and ensure compatibility with the project's ML libraries, we recommend managing Python and your virtual environment using `uv`, an extremely fast Python project manager.

**1. Install uv**
Run the official standalone installer script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

*Note: You will need to restart your terminal or run `source ~/.bashrc` for the `uv` command to be recognized.*

**2. Install Python 3.10**
Instead of building from source, `uv` instantly fetches pre-compiled Python binaries:

```bash
uv python install 3.10

```

**3. Create the Virtual Environment**
Navigate to the root directory of this repository and tell `uv` to create a virtual environment explicitly locked to Python 3.10:

```bash
cd pi-edge-smart-display
uv venv --python 3.10

```

**4. Activate the Environment**
Activate the new virtual environment before running the project or installing the `requirements.txt`:

```bash
source .venv/bin/activate

```

*Verify the installation by running `python --version`. It should output a Python 3.10.x version.*

---

*(You can drop this right into your README above the **Testing** section. Just let me know if you also need the dependency installation step (`uv pip install -r requirements.txt`) added in!)*

---


Here is the complete **Installation** section updated to include the Ollama setup and the specific `qwen2.5:1.5b` model download, matching the default parameters in your `llm_node.py` file.

You can drop this right in place of the previous Installation section.

---

### LLM Setup (Ollama & Qwen)

Since this project runs the reasoning engine entirely on-device, you will need to install Ollama and download the specific language model used in the orchestration backend.

**1. Install Ollama**
Run the official Linux installation script. This will automatically detect your Raspberry Pi's ARM64 architecture and set up Ollama as a background service:

```bash
curl -fsSL https://ollama.com/install.sh | sh

```

**2. Download the Qwen Model**
The Python backend (`src/llm/llm_node.py`) is configured to use the Qwen 2.5 1.5B model by default. Pull this model to your local hardware:

```bash
ollama pull qwen2.5:1.5b

```

*Note: If you plan on running the backend hardware integration tests via `pytest`, the Ollama service must be running in the background and this specific model must be pulled first to avoid a `ResponseError`.*


## Testing

This project uses `pytest` for backend unit and integration testing. The tests are configured to ensure logical isolation and prevent collisions with global system environments.

### Running Tests

**1. Run Fast Unit Tests Only** To run isolated tests that mock hardware and external servers (ideal for rapid development):

```bash
pytest -m "not integration"

```

**2. Run Full Hardware Integration Tests** To run tests that actually trigger the local LLM, GPU, or Audio hardware (requires services like Ollama to be running):

```bash
pytest -m "integration"

```

**3. Run with Debug Logging** By default, `pytest` hides print statements and logs. To view real-time debug outputs and see exactly what the backend is processing, append the following flags:

```bash
pytest -m "not integration" -s --log-cli-level=DEBUG

```

### Troubleshooting: ROS 2 Environment Collisions

Because this project's virtual environment relies on strict pathing, having a global **ROS 2** environment sourced in your terminal (e.g., `/opt/ros/jazzy/`) can cause `pytest` to crash. ROS aggressively injects its own plugins (like `launch_testing`) into the system path, which lack the necessary dependencies inside this project's `.venv`.

**The built-in fix:**
The included `pytest.ini` file is configured to lock down the import paths and prevent wandering:

```ini
[pytest]
pythonpath = .
addopts = --import-mode=importlib
markers =
    integration: marks tests as integration tests that require the Ollama server
```

**The manual override:**
If the `pytest.ini` isolation fails and ROS 2 is still causing `ModuleNotFoundError` crashes before the tests even start, temporarily blind Python to the global environment variables by running your tests with an empty `PYTHONPATH`:

```bash
PYTHONPATH="" pytest

```

*(Tip: Create a terminal alias like `alias pycl="PYTHONPATH=\"\" pytest"` for convenience if you frequently develop with ROS 2 active).*
