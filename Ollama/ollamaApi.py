import requests
import json


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def list_models(self):
        """Get list of available models"""
        res = requests.get(f"{self.base_url}/api/tags")
        res.raise_for_status()
        return [model['name'] for model in res.json().get("models", [])]

    def pull_model(self, model):
        """Pull model from Ollama Hub"""
        res = requests.post(f"{self.base_url}/api/pull", json={"name": model})
        res.raise_for_status()
        for line in res.iter_lines():
            if line:
                yield json.loads(line.decode("utf-8"))

    def chat(self, model, prompt, stream=False):
        """Chat interface"""
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "stream": stream}

        res = requests.post(f"{self.base_url}/api/chat",
                            json=payload, stream=stream)
        res.raise_for_status()

        if stream:
            def stream_generator():
                for line in res.iter_lines():
                    if line:
                        try:
                            yield json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue
            return stream_generator()
        else:
            return res.json()

    def embeddings(self, model, prompt):
        """Generate embeddings from a model"""
        payload = {"model": model, "prompt": prompt}
        res = requests.post(f"{self.base_url}/api/embeddings", json=payload)
        res.raise_for_status()
        return res.json()
