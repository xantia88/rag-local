import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema.document import Document
import json
import llm
from logger import get_logger


def translate(model, content, data):
    context = ("Используй следующие термины:"
               "\n\n"
               f"{content}"
               "\n\n"
               "Cоставь текстовое описание отклонения на русском языке."
               "Используй только ASCII символы в тексте."
               "Исправь орфографические ошибки в тексте."
               "Выведи ответ в один абзац.")
    return llm.request(model, context, data)


def load_documents(model, content_file, terms_file, log):
    documents = []
    with open(content_file, "r") as file:
        objects = json.load(file)
        terms = Path(terms_file).read_text()
        for object in objects:
            text = translate(model, terms, str(object))
            log.info(f"[TRANSLATE] {text}")
            document = [Document(page_content=text)]
            documents.extend(document)
    return documents


def save_documents(documents, filepath):
    with open(filepath, "w") as file:
        for document in documents:
            file.write(document.page_content)
            file.write("\n")


if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

    # load environment variables
    load_dotenv()

    # initialize LLM object
    model = llm.get_model(os.environ)

    # load documents
    content_file = "documents/issues.json"
    terms_file = "config/terms-iss.txt"
    documents = load_documents(model, content_file, terms_file, log)
    log.info(f"{len(documents)} documents loaded")

    # save documents
    file = "content/issues.txt"
    save_documents(documents, file)
    log.info(f"saved to {file}")
