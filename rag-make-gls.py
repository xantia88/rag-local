import os
from pathlib import Path
from dotenv import load_dotenv
from logger import get_logger
import importlib


def translate(model, data):
    context = ("Ты - русскоязычный ассистент. "
               "В документе содержится глоссарий терминов. "
               # "Для каждого термина, который указан в глоссарии, сформируй JSON документ. "
               # "В качестве названия параметров в JSON документе используй названия параметров, которые указанны в глоссарии. "
               "Для каждого термина, который указан в глоссарии, сформируй текстовое описание на русском языке. "
               "Описание должно включать все параметры, которые указаны в глоссарии для данного термина. "
               # "В качестве названия параметров в JSON документе используй названия параметров, которые указанны в глоссарии. "
               "Исправь орфографические ошибки в тексте. "
               "Используй только ASCII символы. "
               # "Выведи ответ в виде массива JSON документов. "
               "Выведи ответ ввиде списка абзацев, разделенных пустой строкой."
               )
    return llm.request(model, context, data)


if __name__ == "__main__":

    # create logger
    log = get_logger("app", __file__)

    # load environment variables
    load_dotenv()

    # import llm module
    name = os.environ.get("llm.module")
    llm = importlib.import_module(name)

    # initialize LLM object
    model = llm.get_model(os.environ)

    # load documents
    content_file = "documents/glossary-test.txt"
    data = Path(content_file).read_text()

    response = translate(model, data)
    print(response)
