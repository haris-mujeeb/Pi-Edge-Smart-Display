import logging
from typing import Any

import ollama

logger = logging.getLogger(__name__)

def get_response_from_ollama(model: str ='qwen2.5:1.5b', prompt: str ="Why is the sky blue?") -> str:
  """Call ollama model and return its text output."""
  logger.info("Starting ollama call: model=%s prompt=%s", model, prompt)

  try:
    resp: Any = ollama.generate(model=model, prompt=prompt)
    logger.debug("Raw response: %s", resp)
    
    if isinstance(resp, dict):
      text = resp.get("text") or resp.get("response") or resp.get("output")
      result = text if text is not None else str(resp)
    else:
      result = str(resp)
      
    logger.info("Ollama call successful")
    return result
    
  except ollama.ResponseError as e:
    logger.error(f"OLLAMA ERROR: The model likely isn't downloaded. Try running 'ollama pull {model}'. Exact error: {e}")
    raise
  
  except Exception as e:
    logger.exception("CRITICAL ERROR during Ollama generation")
    raise