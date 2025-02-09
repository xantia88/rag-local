# rag-local

1. Download ollama application from https://ollama.com/download

2. Install llama3 LLM and Nomic embeddings models

```
ollama pull llama3
ollama pull nomic-embed-text
```

3. View installed models

```
ollama list
```

4. Clone this repo

```
git clone ...
cd rag-local
```

5. Create python virtual environment

```
python3 -m venv .venv
```

6. Activate python virtual environment

```
source .venv/bin/activate
```

7. Install dependencies

```
(.venv) pip install -r requirements.txt
```

