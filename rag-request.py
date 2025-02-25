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

warnings.filterwarnings("ignore")

search_config = {
    "semantic": 4,
    "text": 4,
    "weights": [0.5, 0.5]
}

if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

    # check arguments
    if (len(sys.argv) > 1):

        # load environment variables
        load_dotenv()

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
        vanilla = db.as_retriever(
            search_type="similarity", search_kwargs={"k": search_config["semantic"]})

        # B25 Match
        chunks = db.get()["documents"]
        b25m = BM25Retriever.from_texts(chunks, k=search_config["text"])

        # create retriever
        ensemble = EnsembleRetriever(
            retrievers=[vanilla, b25m], weights=search_config["weights"])

        # read prompt file
        question = Path(sys.argv[1]).read_text()
        log.info(f"[QUESTION] {question}")

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
