import logging
from os import name
import sys
from pathlib import Path

from src.audio.audio_to_text import record_and_save, transcribe_audio
from src.llm.llm_node import get_response_from_ollama

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(audio_filename, audio_output_folder):
  try:
    
    record_and_save(filename=audio_filename, output_folder=audio_output_folder, duration=5)
    
    text = transcribe_audio(file_path=str(filepath))  
  
    logger.info(f"Transcribed Audio: {text}")
    
    response =  get_response_from_ollama(prompt=text)
    
    logger.info(f"Response from LLM: {response}")
    
  except Exception as e:
    logger.exception(f"Pipeline failed: {e}")
    

if __name__ == "__main__":
  audio_output_folder = "audio_files"
  filename = "recoding.wav"
  save_dir = Path(audio_output_folder)
  filepath = save_dir / filename

  run_pipeline(audio_filename=filename, audio_output_folder=audio_output_folder)