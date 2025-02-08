import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_content
import llm

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # load content
    content = load_content("content")

    # split content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(content)

    print(len(documents), "chunks created")

    # load environment variables
    load_dotenv()

    # create embeddings
    db = Chroma.from_documents(
        documents=documents, embedding=llm.get_embeddings(os.environ))

    # initialize LLM object
    model = llm.get_model(os.environ)

    # request / response
    qa_chain = RetrievalQA.from_chain_type(model, retriever=db.as_retriever())

    # request
    question = ("Сформируй список внешних систем."
                "Отвечай на русском языке.")
    print("[QUESTION]", question)

    # response
    response = qa_chain({"query": question})
    print("[ANSWER]", response["result"])
