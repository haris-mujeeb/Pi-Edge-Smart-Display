import logging
from re import DEBUG
from unittest.mock import patch

import ollama
import pytest

from src.llm.llm_node import get_response_from_ollama

logger = logging.getLogger(__name__)


@patch('src.llm.llm_node.ollama.generate')
def test_parses_ollama_dictionary_response_correctly(mock_generate):
  """Test that the function correctly extracts the 'response' key from the dictionary."""
  mock_generate.return_value = {"model": "qwen", "response": "Mocked sky is blue", "done": True}
  
  # Call our function (it will hit the mock, not the real server)
  result = get_response_from_ollama()
  
  assert result == "Mocked sky is blue"
  mock_generate.assert_called_once()

@patch('src.llm.llm_node.ollama.generate')
def test_raises_error_when_model_missing(mock_generate):
  """Test that an ollama.ResponseError is properly caught and re-raised."""
  mock_generate.side_effect = ollama.ResponseError("model 'dummy_model' not found")
  
  with pytest.raises(ollama.ResponseError):
    get_response_from_ollama(model="dummy_model")

@patch('src.llm.llm_node.ollama.generate')
def test_handles_unexpected_dictionary_structure(mock_generate):
  """Test that if the expected keys are missing, it returns the raw string."""  
  weird_response = {"weird_key": "some_value", "status": 200}
  mock_generate.return_value = weird_response
  
  result = get_response_from_ollama()
  
  assert result == str(weird_response)


@pytest.mark.integration
def test_get_response_from_actual_ollama_returns_valid_string():
  """
  Integration test to verify that the Ollama LLM node connects 
  successfully and returns a valid, non-empty string.
  """
  
  prompt = "Hello. Hello."
  resp = get_response_from_ollama(prompt=prompt)
  logger.info(f"Prompt to LLM: {prompt}"
    f"Response from LLM: {resp}")
  
  assert isinstance(resp, str), f"Expected response to be a string, but got {type(resp)}"
  assert len(resp.strip()) > 0, "Expected response to contain text, but got an empty string"