import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from logger import get_logger
import importlib


warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

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
        search_type="similarity", search_kwargs={"k": 4})

    # B25 Match
    chunks = db.get()["documents"]
    b25m = BM25Retriever.from_texts(chunks, k=4)

    # create retriever
    ensemble = EnsembleRetriever(
        retrievers=[vanilla, b25m], weights=[0.5, 0.5])

    # request / response
    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=ensemble)

    # request
    question = "что такое шлюз фиас ?"
    log.info(f"[QUESTION] {question}")

    # retrieve documents
    documents = ensemble.invoke(question)
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
