# Install

1. Download ollama application from https://ollama.com/download

2. Install llama3 LLM and Nomic embeddings models

```
ollama pull llama3.2:3b
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
mkdir logs embeddings
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

Run scripts in any order to transform JSON source documents into text documents using relevant context files. 

```
(.venv) python <script>
```

|Source|Script|Context|Output|
|------|----|-------|------|
|documents/standards.json|rag-make-std.py|config/terms-std.txt|content/standards.txt|
|documents/requirements.json|rag-make-req.py|config/terms-req.txt|content/requirements.txt|
|documents/systems.json|rag-make-sys.py|config/terms-sys.txt|content/systems.txt|
|documents/issues.json|rag-make-iss.py|config/terms-iss.txt|content/issues.txt|


3. Create embeddings and save them in **embeddings/** folder

```
(.venv) python rag-embeddings.py
```

4. Symantic search

```
(.venv) python rag-query.py -f prompts/test.txt -m symantic 
```

5. Text search

```
(.venv) python rag-query.py -f prompts/test.txt -m text
```

6. Perform RAG based request to LLM

```
(.venv) python3 rag-request.py prompts/test.txt
```

# Troubleshooting

See **logs/** for error messages.

