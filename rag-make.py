import os
from pathlib import Path
from dotenv import load_dotenv
import warnings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
import json

warnings.filterwarnings("ignore")


def request(llm, context, data):

    messages = [
        SystemMessage(context),
        HumanMessage(data)
    ]

    response = llm.invoke(messages)
    parser = StrOutputParser()
    text = parser.invoke(response)
    return text


def translate(llm, content, data):
    context = (f"Используй следующие термины: {content}."
               "Cоставь краткое текстовое описание системы на русском языке."
               "Выведи описание в один абзац.")
    return request(llm, context, data)


def load_documents(content_file, terms_file):
    documents = []
    with open(content_file, "r") as file:
        systems = json.load(file)
        terms = Path(terms_file).read_text()
        for system in systems:
            text = translate(llm, terms, str(system))
            print("[TRANSLATE]", text)
            document = [Document(page_content=text)]
            documents.extend(document)
    return documents


def save_documents(documents, filepath):
    with open(filepath, "w") as file:
        for document in documents:
            file.write(document.page_content)
            file.write("\n")


if __name__ == "__main__":

    # load environment variables
    load_dotenv()

    # initialize LLM object
    llm = ChatOllama(
        model=os.environ["model"],
        temperature=0,
    )

    # load documents
    content_file = "documents/systems.json"
    terms_file = "config/terms.txt"
    documents = load_documents(content_file, terms_file)
    print(len(documents), "documents loaded")

    # save documents
    file = "content/systems.txt"
    save_documents(documents, file)
    print("saved to", file)
