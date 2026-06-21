import logging
import os

import numpy as np

from src.audio.audio_to_text import record_and_save

logger = logging.getLogger(__name__)

def test_record_and_save_creates_wav(tmp_path):
  """
  Integration test to verify the physical microphone captures real audio.
  NOTE: This test requires a working microphone and OS permissions. 
  It will likely fail in automated CI/CD pipelines (like GitHub Actions).
  """

  test_filename = "test_hardware_mic.wav"
  duration = 0.1
  
  if os.path.exists(test_filename):
    os.remove(test_filename)
  
  logger.info(f"Recoding for {duration} seconds.")
  recording = record_and_save(
    filename=test_filename, 
    output_folder=tmp_path, 
    duration=duration
    )

  assert recording is not None
  assert recording.size > 0
  
  assert os.path.exists((tmp_path / test_filename))
  
  rms_energy = np.sqrt(np.mean(recording**2))
  logger.info(f"Microphone recorded silence (RMS: {rms_energy}). ")
  assert rms_energy > 0.00001, ("Check your microphone hardware and OS privacy permissions.")
  
  if os.path.exists(test_filename):
    os.remove(test_filename)