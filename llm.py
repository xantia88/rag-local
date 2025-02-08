
from langchain_ollama import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


def get_model(env):
    return ChatOllama(
        model=env.get("model"),
        temperature=0,
    )


def get_embeddings(env):
    return SentenceTransformerEmbeddings(model_name=env.get("embeddings"))


def request(llm, context, data):

    messages = [
        SystemMessage(context),
        HumanMessage(data)
    ]

    response = llm.invoke(messages)
    parser = StrOutputParser()
    text = parser.invoke(response)
    return text


if __name__ == "__main__":
    exit()
