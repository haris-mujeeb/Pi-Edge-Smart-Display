import logging
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

def get_i2c_mic_id() -> int:
  """Scans audio devices and automatically finds the Google VoiceHAT I2C mic"""
  try:
    devices = sd.query_devices()
    for i, device in enumerate(devices):
      name = device['name'].lower()
      if ('googlevoicehat' in name or 'voicehat' in name) and device['max_input_channels'] > 0:
        logger.info(f"Auto-detected I2S Microphone at index {i}: {device['name']}")
        return i
      
  except Exception as e:
    logger.error(f"Error querrying audio devices: {e}")
  
  logger.warning("Could not auto-detect I2S microphone. Falling back to default.")
  return None


def record_and_save(filename="recording.wav", output_folder="audio_files", duration=2.0, samplerate=48000, channels=1, device_id=None):
  """Record audio and save to a WAV file in a specific folder"""
  
  if device_id is not None:
    sd.default.device = device_id
  
  sd.default.samplerate = samplerate
  sd.default.channels = channels

  save_dir = Path(output_folder)
  save_dir.mkdir(parents=True, exist_ok=True)
  filepath = save_dir / filename

  logger.info(f"Recording {duration} s at {samplerate} Hz, {channels} channels...")

  try:
    recording = sd.rec(int(duration * samplerate), dtype='float32')
    sd.wait()
    logger.info("Recording finished")
    
    if channels == 2 and recording.ndim == 2:    
      # Example: convert to mono by averaging channels
      mono = np.mean(recording, axis=1)
      sf.write(filepath, mono, samplerate)
    else:
      sf.write(filepath, recording, samplerate)

    logger.info(f"Saved to {filepath}")
    return recording

  except Exception as e:
    logger.error(f"Failed to record: {e}")
    logger.info("Tip: Run python3 -m sounddevice to list available audio devices.")
    return None
  

def transcribe_audio(file_path : str) -> str:
  logger.info("Loading tiny.en model...")
  model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
  
  start_time = time.time()
  logger.info("Transcribing")
  segments, info = model.transcribe(file_path, beam_size=1)
  
  text_segments = []
  
  for segment in segments:
    logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    text_segments.append(segment.text.strip())

  full_text = " ".join(text_segments)
  
  elapsed = time.time() - start_time
  logger.info(f"Done! Processed in {elapsed:.2f}s")
  
  return full_text
