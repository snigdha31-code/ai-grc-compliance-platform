import requests

OLLAMA_URL = "http://127.0.0.1:11434" # Ollama server URL
MODEL_NAME = "phi3" # Name of the model we want to use in Ollama. 
#This should match the name in your ollama.yaml config.

# send prompt to Ollama and return the response text.
# Raises exception on error.
def ask_ollama(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]
