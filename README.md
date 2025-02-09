# Install

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

# Run

1. Activate python virtual environment

```
cd rag-local
source .venv/bin/activate
```

2. Prepare data and store it as text files in **content/** folder

2.1. Systems

```
(.venv) python3 rag-make-sys.py
```

2.2. Standards

```
(.venv) python3 rag-make-std.py
```

3. Create embeddings and save them in **embeddings/** folder

```
(.venv) python3 rag-embeddings.py
```

4. Perform RAG based request to LLM

```
(.venv) python3 rag-request.py
```

# Troubleshooting

See **logs/** for error messages.

