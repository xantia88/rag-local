from os import listdir, path
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.schema.document import Document


def load_lines(file):
    text = Path(file).read_text()
    lines = text.split("\n")
    documents = []
    for line in lines:
        documents.append(Document(page_content=line,
                         metadata={"source": file}))
    return documents


def load_text(file):
    loader = TextLoader(file)
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
                docs = load_lines(file)
            elif extension == ".csv":
                docs = load_text(file)
            documents.extend(docs)

    return documents


def merge(documents):
    arr = []
    for doc in documents:
        arr.append(doc.page_content)
    return [Document(page_content="\n\n".join(arr))]


if __name__ == "__main__":
    exit()
