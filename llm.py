from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from logger import get_silent_logger

log = get_silent_logger("llm", __file__)


def get_model(env):
    return ChatOllama(
        model=env.get("model"),
        temperature=0,
    )


def get_embeddings(env):
    return OllamaEmbeddings(model=env.get("embeddings"))


def request(llm, context, data):

    messages = [
        SystemMessage(context),
        HumanMessage(data)
    ]

    log.info(f"[SYSTEM] {context}")
    log.info(f"[HUMAN] {data}")

    response = llm.invoke(messages)
    parser = StrOutputParser()
    text = parser.invoke(response)
    log.info(f"[RESPONSE] {text}")

    return text


if __name__ == "__main__":
    exit()
