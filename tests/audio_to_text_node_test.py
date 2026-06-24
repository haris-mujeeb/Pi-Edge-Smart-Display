import logging
import os

from pathlib import Path
import numpy as np

from src.audio.audio_to_text_node import get_i2c_mic_id, record_and_save, transcribe_audio

logger = logging.getLogger(__name__)

def test_mic_record_and_creates_wav(tmp_path):
  """
  Integration test to verify the physical microphone captures real audio.
  NOTE: This test requires a working microphone and OS permissions. 
  It will likely fail in automated CI/CD pipelines (like GitHub Actions).
  """

  test_filename = "test_hardware_mic.wav"
  duration = 0.1
  
  if os.path.exists(test_filename):
    os.remove(test_filename)
  
  mic_id = get_i2c_mic_id()
  
  logger.info(f"Recording for {duration} seconds.")
  recording = record_and_save(
    filename=test_filename, 
    output_folder=tmp_path, 
    duration=duration,
    samplerate=48000,
    channels=2,
    device_id=mic_id
    )

  assert recording is not None
  assert recording.size > 0
  
  assert os.path.exists((tmp_path / test_filename))
  
  rms_energy = np.sqrt(np.mean(recording**2))
  logger.info(f"Microphone recorded silence (RMS: {rms_energy}). ")
  assert rms_energy > 0.00001, ("Check your microphone hardware and OS privacy permissions.")
  
  if os.path.exists(test_filename):
    os.remove(test_filename)
    
def test_audio_to_text():
  """
  
  """
  test_file_dir = "audio_files"
  test_filename = "hello.wav"
  test_filepath = Path(test_file_dir) / test_filename
  
  assert os.path.exists(test_filepath)
  
  text = transcribe_audio(file_path=str(test_filepath))
  
  logger.info(f"Response: {text}")