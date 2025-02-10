import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from logger import get_logger
import llm

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

    # load environment variables
    load_dotenv()

    # create embeddings and save to local filesystem
    dir = "embeddings"
    embeddings = llm.get_embeddings(os.environ)
    db = Chroma(persist_directory=dir, embedding_function=embeddings)

    # initialize LLM object
    model = llm.get_model(os.environ)

    # request / response
    qa_chain = RetrievalQA.from_chain_type(
        model, retriever=db.as_retriever(search_kwargs={"k": 4}))

    # request
    question = ("что такое шлюз фиас? "
                "Отвечай на русском языке.")
    log.info(f"[QUESTION] {question}")

    # response
    response = qa_chain({"query": question})
    answer = response["result"]
    log.info(f"[ANSWER] {answer}")
