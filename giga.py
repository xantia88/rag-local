
from langchain_gigachat.chat_models import GigaChat
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from logger import get_silent_logger

log = get_silent_logger("llm", __file__)


def get_model(env):
    return GigaChat(
        credentials=env.get("giga.auth_key"),
        scope=env.get("giga.scope"),
        model=env.get("giga.model"),
        verify_ssl_certs=False,
    )


def get_embeddings(env):
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


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
