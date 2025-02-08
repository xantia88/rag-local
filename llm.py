
from langchain_ollama import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings


def get_model(env):
    return ChatOllama(
        model=env.get("llm_model"),
        temperature=0,
    )


def get_embeddings(env):
    return SentenceTransformerEmbeddings(model_name=env.get("embeddings_model"))


if __name__ == "__main__":
    exit()
