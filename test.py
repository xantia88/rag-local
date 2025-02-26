from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize

retriever = BM25Retriever.from_texts(
    ["foo", "bar", "world", "hello", "foo bar"], preprocess_func=word_tokenize)
result = retriever.invoke("xxxx")
print(result)
