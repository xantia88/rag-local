import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_content
from langchain_community.vectorstores.utils import filter_complex_metadata
from llm import get_model, get_embeddings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # load content
    data = load_content("content")

    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(data)

    print(len(documents), "chunks created")

    # load environment variables
    load_dotenv()

    # create embeddings and put them into in-memory vector storage
    db = Chroma.from_documents(filter_complex_metadata(
        documents), get_embeddings(os.environ))

    # create prompt
    question = ("Используй русский язык."
                "Выведи список внутренних систем?")

    # initialize LLM object
    llm = get_model(os.environ)

    # request / response
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    print("[QUESTION]", question)
    response = qa_chain({"query": question})
    print("[ANSWER]", response["result"])
