import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


def record_and_save(filename="recoding.wav", output_folder="audio_files", duration=2.0, samplerate=44100, channels=2):
  """Record audio and save to a WAV file in a specific folder"""
  sd.default.samplerate = samplerate
  sd.default.channels = channels

  save_dir = Path(output_folder)
  
  save_dir.mkdir(parents=True, exist_ok=True)
  
  filepath = save_dir / filename

  logger.info(f"Recoding {duration} s at {samplerate} Hz, {channels} channels...")
  recording = sd.rec(int(duration * samplerate), dtype='float32')
  sd.wait()
  logger.info("Recoding finished")
  
  if channels == 2 and recording.ndim == 2:    
    # Example: convert to mono by averaging channels
    mono = np.mean(recording, axis=1)
    sf.write(filepath, mono, samplerate)
  else:
    sf.write(filepath, recording, samplerate)

  logger.info(f"Saved to {filepath}")
  return recording
