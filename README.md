# SuperEasy 100% Local RAG with Ollama + Email RAG

### YouTube Tutorials
- https://www.youtube.com/watch?v=Oe-7dGDyzPM
- https://www.youtube.com/watch?v=vFGng_3hDRk
### Latest YouTube Updated Features
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/0X7raDGDyzPM/0.jpg)](https://www.youtube.com/watch?v=0X7raDGDyzPM)

## Standardized Pipelines

This repository now includes three standardized pipelines for evaluation and comparison:

### 1. RAG Pipeline (`rag_pipeline.py`)
Retrieval-Augmented Generation with query rewriting for multi-turn conversations.
- Uses external knowledge base (vault.txt)
- Rewrites queries based on conversation history
- Retrieves relevant context using embeddings
- Generates answers based on retrieved context

### 2. Chain of Thought Pipeline (`chain_of_thought_pipeline.py`)
Standalone reasoning without external knowledge.
- No retrieval from external sources
- Uses step-by-step reasoning prompts
- Generates answers through logical thinking process

### 3. Baseline Pipeline (`baseline_pipeline.py`)
Direct LLM generation without augmentation.
- No retrieval or reasoning steps
- Direct question-to-answer generation
- Simple baseline for comparison

## Quick Start

### Setup
1. git clone https://github.com/AllAboutAI-YT/easy-local-rag.git
2. cd dir
3. pip install -r requirements.txt
4. Install Ollama (https://ollama.com/download)
5. ollama pull llama3
6. ollama pull mxbai-embed-large

### Run Individual Pipelines

#### RAG Pipeline
```bash
python rag_pipeline.py
```

#### Chain of Thought Pipeline
```bash
python chain_of_thought_pipeline.py
```

#### Baseline Pipeline
```bash
python baseline_pipeline.py
```

### Evaluate All Pipelines
```bash
# Evaluate on HotpotQA subset
python evaluate.py --samples 10

# Evaluate specific pipeline
python evaluate.py --pipeline rag --samples 5
```

## Configuration

All pipelines use `config.yaml` for configuration:
- `vault_file`: Path to knowledge base file
- `embeddings_file`: Path to cached embeddings
- `ollama_model`: LLM model to use
- `top_k`: Number of context chunks to retrieve (RAG only)
- Pipeline-specific settings under `pipelines` section

## Legacy Scripts (Deprecated)

### Email RAG Setup
1. git clone https://github.com/AllAboutAI-YT/easy-local-rag.git
2. cd dir
3. pip install -r requirements.txt
4. Install Ollama (https://ollama.com/download)
5. ollama pull llama3 (etc)
6. ollama pull mxbai-embed-large
7. set YOUR email logins in .env (for gmail create app password (video))
9. python collect_emails.py to download your emails
10. python emailrag2.py to talk to your emails

### Latest Updates
- Added Email RAG Support (v1.3)
- Upload.py (v1.2)
   - replaced /n/n with /n
- New embeddings model mxbai-embed-large from ollama (1.2)
- Rewrite query function to improve retrival on vauge questions (1.2)
- Pick your model from the CLI (1.1)
  - python localrag.py --model mistral (llama3 is default)
- Talk in a true loop with conversation history (1.1)

### My YouTube Channel
https://www.youtube.com/c/AllAboutAI

### What is RAG?
RAG is a way to enhance the capabilities of LLMs by combining their powerful language understanding with targeted retrieval of relevant information from external sources often with using embeddings in vector databases, leading to more accurate, trustworthy, and versatile AI-powered applications

### What is Ollama?
Ollama is an open-source platform that simplifies the process of running powerful LLMs locally on your own machine, giving users more control and flexibility in their AI projects. https://www.ollama.com

### To create the Vault from hotpotQA dataset
python run_create_vault.py
Get-Content -Path vault.txt -TotalCount 10