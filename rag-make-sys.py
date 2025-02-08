import os
from pathlib import Path
from dotenv import load_dotenv
import warnings
from langchain.schema.document import Document
import json
import llm

warnings.filterwarnings("ignore")


def translate(model, content, data):
    context = (f"Используй следующие термины: {content}."
               "Cоставь краткое текстовое описание системы на русском языке."
               "Используй только ASCII символы в тексте."
               "Исправь орфографические ошибки в тексте."
               "Выведи описание в один абзац.")
    return llm.request(model, context, data)


def load_documents(model, content_file, terms_file):
    documents = []
    with open(content_file, "r") as file:
        objects = json.load(file)
        terms = Path(terms_file).read_text()
        for object in objects:
            text = translate(model, terms, str(object))
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
    model = llm.get_model(os.environ)

    # load documents
    content_file = "documents/systems.json"
    terms_file = "config/terms-sys.txt"
    documents = load_documents(model, content_file, terms_file)
    print(len(documents), "documents loaded")

    # save documents
    file = "content/systems.txt"
    save_documents(documents, file)
    print("saved to", file)
