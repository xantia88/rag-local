import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import warnings
from langchain_chroma import Chroma
from logger import get_logger
import importlib

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

    if len(sys.argv) > 1:

        # load environment variables
        load_dotenv()

        # import llm
        name = os.environ.get("llm.module")
        llm = importlib.import_module(name)

        # create embeddings and save to local filesystem
        dir = "embeddings"
        embeddings = llm.get_embeddings(os.environ)
        db = Chroma(persist_directory=dir, embedding_function=embeddings)

        # retrieve data from vectorstore
        question = Path(sys.argv[1]).read_text()
        documents = db.similarity_search(question, k=5)
        for document in documents:
            print(document)
            print("---")

    else:
        log.error("No prompt file provided")
