import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
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

    # maximum marginal relevance
    mmr = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 4})

    # create retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vanilla, mmr], weights=[0.5, 0.5])

    # request / response
    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=ensemble_retriever)

    # request
    question = ("что такое шлюз фиас? "
                "Отвечай на русском языке.")
    log.info(f"[QUESTION] {question}")

    # response
    response = qa_chain({"query": question})
    answer = response["result"]
    log.info(f"[ANSWER] {answer}")
