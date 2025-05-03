from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


def run_chatbot(model_name='deepcoder'):
    # Initialize the Ollama LLM
    llm = Ollama(model=model_name)

    # Memory to hold conversation history
    memory = ConversationBufferMemory()

    # Create a conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # set to False to hide internal chain logs
    )

    print(
        f"\n🧠 Chatbot running with Ollama model '{model_name}'. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
        response = conversation.predict(input=user_input)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    run_chatbot()
