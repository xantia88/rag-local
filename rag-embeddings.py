import os
from dotenv import load_dotenv
import warnings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_content
from logger import get_logger
import importlib

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

    # load content
    content = load_content("content")
    log.info(f"{len(content)} documents loaded")
    for document in content:
        log.info(document.metadata.get("source"))

    # split content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(content)
    log.info(f"{len(documents)} chunks created")

    # load environment variables
    load_dotenv()

    # import llm
    name = os.environ.get("llm.module")
    llm = importlib.import_module(name)

    # create embeddings and save to local filesystem
    dir = "embeddings"
    embeddings = llm.get_embeddings(os.environ)
    db = Chroma.from_documents(persist_directory=dir,
                               documents=documents,
                               embedding=embeddings)
    log.info(f"{len(documents)} chunks saved in {dir}")
