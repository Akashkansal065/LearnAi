import ollama


def get_ollama_models_list():
    """Fetches a list of model names available in the local Ollama instance."""
    try:
        models_data = ollama.list()
        return [model.model for model in models_data.models]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}. Returning empty list.")
        return []


print(get_ollama_models_list())
