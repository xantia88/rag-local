import os
from pathlib import Path
from dotenv import load_dotenv
import warnings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
import json
from llm import get_model

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
               "Cоставь текстовое описание требования на русском языке."
               "Текст ответа должен содержать уникальный идентификатор требования, текст требования, версию стандарта и дополнительные параметры требования."
               "Используй только ASCII символы в тексте."
               "Исправь орфографические ошибки в тексте."
               "Выведи ответ в один абзац.")
    return request(llm, context, data)


def load_documents(llm, content_file, terms_file):
    documents = []
    with open(content_file, "r") as file:
        objects = json.load(file)
        terms = Path(terms_file).read_text()
        for object in objects:
            text = translate(llm, terms, str(object))
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
    llm = get_model(os.environ)

    # load documents
    content_file = "documents/standards.json"
    terms_file = "config/terms-std.txt"
    documents = load_documents(llm, content_file, terms_file)
    print(len(documents), "documents loaded")

    # save documents
    file = "content/standards.txt"
    save_documents(documents, file)
    print("saved to", file)
