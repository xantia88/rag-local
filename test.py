from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

if __name__ == "__main__":

    load_dotenv()

    llm = ChatOllama(
        model=os.environ["model"],
        temperature=0,
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to Russian. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)

    print(ai_msg.content)
