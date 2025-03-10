from langchain.prompts import ChatPromptTemplate
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import warnings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema.document import Document
from logger import get_logger
import importlib
import json

warnings.filterwarnings("ignore")

search_config = {
    "semantic": 10,
    "threshold": 0.5,
    "text": 3,
    "weights": [0.7, 0.3]
}


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

    # check arguments
    if (len(sys.argv) > 1):

        # load environment variables
        load_dotenv()

        # read prompt file
        question, sources = get_request_data(sys.argv[1])
        log.info(f"[QUESTION] {question}")
        log.info(f"[SOURCES] {sources}")

        # import llm
        name = os.environ.get("llm.module")
        llm = importlib.import_module(name)

        # load embeddings
        dir = "embeddings"
        embeddings = llm.get_embeddings(os.environ)
        db = Chroma(persist_directory=dir, embedding_function=embeddings)

        # semantic search parameters
        args = {}
        args["k"] = search_config["semantic"]
        args["score_threshold"] = search_config["threshold"]
        if sources is not None:
            args["filter"] = {
                "source": {
                    "$in": sources
                }
            }

        # retrieve relevant documents via semantic search
        semantic = db.as_retriever(
            search_type="similarity_score_threshold", search_kwargs=args)
        relevant_documents = semantic.invoke(question)

        log.info(
            f"{len(relevant_documents)} relevant documents found")

        if (len(relevant_documents) > 0):

            # extract sources
            relevant_sources = {doc.metadata['source'] 
                                for doc in relevant_documents}
            relevant_sources = list(relevant_sources)

            log.info(f"{relevant_sources} - relevant sources")

            # text search parameters
            where_clause = {
                "source": {
                    "$in": relevant_sources
                }
            }

            # get all chunks for the sources of relevant documents
            collection = db.get(
                where=where_clause, include=["documents"])
            chunks = [Document(page_content=text) 
                      for text in collection['documents']]

            # create text retriever
            bm25 = BM25Retriever.from_documents(
                chunks, k=search_config["text"])

            # create ensemble retriever
            ensemble = EnsembleRetriever(
                retrievers=[semantic, bm25], weights=search_config["weights"])

            # retrieve documents
            documents = ensemble.invoke(question)
            log.info(f"{len(documents)} documents found via ensemble")

            # extract texts
            texts = [document.page_content for document in documents]
            context_text = "\n\n".join(texts)

            # execute prompt
            prompt_template = """
                Для ответа на вопрос используй только следующий контекст:

                {context}

                ---
                Ответь на вопрос: {query}
                Для ответа используй русский язык.
                """
            prompt_template = ChatPromptTemplate.from_template(prompt_template)
            prompt = prompt_template.format(
                context=context_text, query=question)
            log.info("start request")
            model = llm.get_model(os.environ)
            response = model.invoke(prompt)
            print(response.content)
            log.info("stop request")

    else:
        log.error("No prompt file provided")
