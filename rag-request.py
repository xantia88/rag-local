import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import warnings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from logger import get_logger
import importlib
import json

warnings.filterwarnings("ignore")

search_config = {
    "semantic": 4,
    "text": 4,
    "weights": [0.5, 0.5]
}


def get_request_data(file):
    if file.endswith(".txt"):
        return Path(file).read_text(), None
    elif file.endswith(".json"):
        with open(file, "r") as file:
            request = json.load(file)
            file = request["text"]
            return Path(file).read_text(), request["source"]


if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

    # check arguments
    if (len(sys.argv) > 1):

        # load environment variables
        load_dotenv()

        # read prompt file
        question, sources = get_request_data(sys.argv[1])
        log.info(f"[QUESTION] {question}")
        log.info(f"[SOURCES] {sources}")

        # import llm
        name = os.environ.get("llm.module")
        llm = importlib.import_module(name)

        # create embeddings and save to local filesystem
        dir = "embeddings"
        embeddings = llm.get_embeddings(os.environ)
        db = Chroma(persist_directory=dir, embedding_function=embeddings)

        # initialize LLM object
        model = llm.get_model(os.environ)

        # similarity search
        args = {"k": search_config["semantic"]}
        args["score_threshold"] = 0.5
        if sources is not None:
            args["filter"] = {
                "source": {
                    "$in": sources
                }
            }
        vanilla = db.as_retriever(
            search_type="similarity_score_threshold", search_kwargs=args)

        # B25 Match
        chunks = db.get()["documents"]
        b25m = BM25Retriever.from_texts(chunks, k=search_config["text"])

        # create retriever
        ensemble = EnsembleRetriever(
            retrievers=[vanilla, b25m], weights=search_config["weights"])

        # retrieve documents
        documents = ensemble.invoke(question)
        print(f"{len(documents)} found")
        texts = [document.page_content for document in documents]
        context = "\n\n".join(texts)

        # generate answer with LLM
        context = (f"Используй следующий контекст:"
                   "\n\n"
                   f"{context}"
                   "\n\n"
                   "Ответь на вопрос пользователя на русском языке.")
        answer = llm.request(model, context, question)
        log.info(f"[ANSWER] {answer}")

    else:
        log.error("No prompt file provided")
