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

This repository follows a modular, service-oriented architecture to separate hardware interfaces, AI models, and the web frontend.

```text
pi-edge-smart-display/
├── hardware/               # Hardware configuration and testing scripts
│   ├── asound.conf         # ALSA audio configuration for I2S shared clock
│   └── boot_config.txt     # Pi /boot/firmware/config.txt overlays
├── models/                 # Local directory for downloaded weights (Git-ignored)
│   ├── llm/                # GGUF files for llama.cpp/Ollama
│   ├── vision/             # .tflite models compiled for Edge TPU
│   └── voice/              # Whisper and Piper model files
├── src/                    # Core Python backend and orchestration
│   ├── api/                # FastAPI application and routes
│   ├── audio/              # I2S input/output, wake word, and TTS managers
│   ├── llm/                # Prompts, function calling, and Ollama integration
│   ├── vision/             # Coral TPU face detection and camera loop
│   └── main.py             # Main execution loop and service manager
├── tests/                  # Unit and integration tests (gtest/pytest)
│   ├── test_audio.py
│   └── test_vision.py
├── ui/                     # Frontend web application
│   ├── public/             
│   ├── src/                # React/Vue components (Dashboard, Chat, widgets)
│   ├── package.json
│   └── vite.config.js
├── scripts/                # Utility scripts (setup, install, run)
│   ├── setup_env.sh        # Installs dependencies and apt packages
│   └── start_kiosk.sh      # Launches Chromium in full-screen kiosk mode
├── .gitignore
├── requirements.txt        # Python dependencies
└── README.md
