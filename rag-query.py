import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import warnings
from langchain_chroma import Chroma
from logger import get_logger
import importlib
from langchain_community.retrievers import BM25Retriever
import json

warnings.filterwarnings("ignore")

search_config = {
    "semantic": 4,
    "text": 4
}


def get_param(name):
    if name in sys.argv:
        pos = sys.argv.index(name)
        return sys.argv[pos+1]
    return None


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

    if len(sys.argv) > 4:

        search_mode = get_param("-m")
        prompt_file = get_param("-f")

        if search_mode != None and prompt_file != None:

            # load environment variables
            load_dotenv()

            # import llm
            name = os.environ.get("llm.module")
            llm = importlib.import_module(name)

            # create embeddings and save to local filesystem
            dir = "embeddings"
            embeddings = llm.get_embeddings(os.environ)
            db = Chroma(persist_directory=dir, embedding_function=embeddings)

            # prepare request
            question, sources = get_request_data(prompt_file)

            # quesry
            if search_mode in search_config:
                limit = search_config[search_mode]
                # semantic search
                if search_mode == "semantic":

                    args = {"k": limit}
                    if sources is not None:
                        args["filter"] = {
                            "source": {
                                "$in": sources
                            }
                        }

                    retriever = db.as_retriever(search_kwargs=args)
                    for document in retriever.invoke(question):
                        print(document)
                        print("---")
                # text search
                elif search_mode == "text":
                    chunks = db.get()["documents"]
                    b25m = BM25Retriever.from_texts(chunks, k=limit)
                    for document in b25m.invoke(question):
                        print(document)
                        print("---")
            else:
                log.error(
                    f"Unknown mode: {search_mode}, use {list(search_config.keys())}")

        else:
            log.error(
                f"Use -f to for prompt file ({prompt_file}), -m for search mode ({search_mode})")

    else:
        log.error("Prompt file (-f) and search mode (-m) should be provided")
