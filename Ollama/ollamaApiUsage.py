from ollamaApi import OllamaClient


ollama = OllamaClient()

# ✅ List models
print(ollama.list_models())

# ✅ Pull a model (stream logs)
# for update in ollama.pull_model("llama3"):
#     print(update)

# ✅ Single chat response
res = ollama.chat("phi4", "Explain gravity", stream=False)
print(res["message"]["content"])

# ✅ Streaming chat
for chunk in ollama.chat("phi4", "Tell me a joke", stream=True):
    print(chunk["message"]["content"], end="", flush=True)

# ✅ Embeddings
# emb = ollama.embeddings("nomic-embed-text", "What is the meaning of life?")
# print(emb)
