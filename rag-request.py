import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_content
from logger import get_logger
import llm


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

    # create embeddings
    db = Chroma.from_documents(
        documents=documents, embedding=llm.get_embeddings(os.environ))
    log.info("put documents to vectorstore")

    # initialize LLM object
    model = llm.get_model(os.environ)

    # request / response
    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=db.as_retriever(search_kwargs={"k": 10}))

    # request
    question = ("Сформируй список внешних систем."
                "Отвечай на русском языке.")
    log.info(f"[QUESTION] {question}")

    # response
    response = qa_chain({"query": question})
    answer = response["result"]
    log.info(f"[ANSWER] {answer}")
