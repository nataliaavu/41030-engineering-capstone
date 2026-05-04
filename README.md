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
- Retrieves relevant context using embeddings
- Generates answers based on retrieved context

### 2. Chain of Thought Pipeline (`chain_of_thought_pipeline.py`)
Standalone reasoning without external knowledge.
- No retrieval from external sources
- Uses step-by-step reasoning prompts
- Generates answers through logical thinking process

### 3. CoT and RAG Pipeline (`cot_and_rag_pipeline.py`)
Combined Chain of Thought reasoning and Retrieval-Augmented Generation.
- Uses external knowledge base (vault.txt) as well as step-by-step reasoning prompts
- Generates answers based on retireved context as well as a logical thinking process

### 4. Baseline Pipeline (`baseline_pipeline.py`)
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
python pipelines/rag_pipeline.py
```

#### Chain of Thought Pipeline
```bash
python pipelines/chain_of_thought_pipeline.py
```

#### COTRAG Pipeline
```bash
python pipelines/cot_and_rag_pipeline.py
```

#### Baseline Pipeline
```bash
python pipelines/baseline_pipeline.py
```

### Evaluate All Pipelines
```bash
# Evaluate exact match
python evaluations/evaluate_em.py --samples 10

# Evaluate F1 score
python evaluations/evaluate_f1.py --samples 10

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

### To create the Vault from hotpotQA dataset
python utils/run_create_vault.py
Get-Content -Path utils/vault.txt -TotalCount 10
### To create the Vault from hotpotQA dataset
python utils/run_create_vault.py
Get-Content -Path utils/vault.txt -TotalCount 10

### What is RAG?
RAG is a way to enhance the capabilities of LLMs by combining their powerful language understanding with targeted retrieval of relevant information from external sources often with using embeddings in vector databases, leading to more accurate, trustworthy, and versatile AI-powered applications

### What is Ollama?
Ollama is an open-source platform that simplifies the process of running powerful LLMs locally on your own machine, giving users more control and flexibility in their AI projects. https://www.ollama.com
