import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import warnings
from langchain_chroma import Chroma
from logger import get_logger
import importlib
from langchain_community.retrievers import BM25Retriever
from langchain.schema.document import Document
import json


warnings.filterwarnings("ignore")

search_config = {
    "semantic": 10,
    "threshold": 0.5,
    "text": 3
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


def show(documents):
    log.info(f"{len(documents)} found")
    for document in documents:
        print(document)
        print("---")


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

                # semantic search
                args = {}
                args["k"] = search_config["semantic"]
                args["score_threshold"] = search_config["threshold"]
                if sources is not None:
                    args["filter"] = {
                        "source": {
                            "$in": sources
                        }
                    }

                retriever = db.as_retriever(
                    search_type="similarity_score_threshold", search_kwargs=args)
                relevant_documents = retriever.invoke(question)

                if search_mode == "semantic":
                    show(relevant_documents)

                # text search
                if search_mode == "text":

                    log.info(
                        f"{len(relevant_documents)} relevant documents found")

                    if (len(relevant_documents) > 0):

                        relevant_sources = {
                            doc.metadata['source'] for doc in relevant_documents}

                        log.info(f"{relevant_sources} - relevant sources")

                        where_clause = {
                            "source": {
                                "$in": list(relevant_sources)
                            }
                        }

                        collection = db.get(
                            where=where_clause, include=["documents"])
                        chunks = [Document(page_content=text)
                                  for text in collection['documents']]

                        print(len(chunks))
                        bm25 = BM25Retriever.from_documents(
                            chunks, k=search_config["text"])
                        documents = bm25.invoke(question)
                        show(documents)

            else:
                log.error(
                    f"Unknown mode: {search_mode}, use {list(search_config.keys())}")

        else:
            log.error(
                f"Use -f to for prompt file ({prompt_file}), -m for search mode ({search_mode})")

    else:
        log.error("Prompt file (-f) and search mode (-m) should be provided")
