from os import listdir, path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredExcelLoader


def load_txt(file):
    loader = TextLoader(file)
    docs = loader.load()
    return docs


def load_excel(file):
    loader = UnstructuredExcelLoader(file, mode="elements")
    docs = loader.load()
    return docs


def load_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load()
    return docs


def load_csv(file):
    loader = CSVLoader(file)
    docs = loader.load()
    return docs


def load_html(file):
    loader = UnstructuredHTMLLoader(file)
    docs = loader.load()
    return docs


def load_content(dir):

    arr = [path.join(dir, file) for file in listdir(
        dir) if path.isfile(path.join(dir, file))]

    documents = []
    for file in arr:
        extension = path.splitext(file)[1]
        if extension:
            extension = extension.lower()
            docs = []
            if extension == ".txt":
                docs = load_txt(file)
            elif extension == ".csv":
                docs = load_txt(file)
            elif extension == ".pdf":
                docs = load_pdf(file)
            elif extension in [".htm", ".html"]:
                docs = load_html(file)
            elif extension in [".xls", ".xlsx"]:
                docs = load_excel(file)
            if len(docs) > 0:
                print("load", file, len(docs))
                documents.extend(docs)

    return documents


if __name__ == "__main__":
    exit()
