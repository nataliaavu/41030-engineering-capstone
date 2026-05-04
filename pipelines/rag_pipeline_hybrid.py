import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import ollama
from openai import OpenAI
import json
import yaml
from rank_bm25 import BM25Okapi
import numpy as np
from utils.run_create_vault import create_vault_from_hotpot

# ANSI escape codes for colors
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def load_config(config_path=None):
    """Load configuration from YAML file"""
    config_path = config_path or ROOT_DIR / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_vault_exists(vault_path, level='hard', n=1000):
    """Ensure a vault file exists by generating it from the Hotpot dataset if needed."""
    vault_path = resolve_repo_path(vault_path)
    if vault_path.exists():
        return vault_path

    print(f"Vault file not found at {vault_path}. Creating vault from Hotpot dataset...")
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    create_vault_from_hotpot(vault_path=vault_path, level=level, n=n)

    if not vault_path.exists():
        raise FileNotFoundError(f"Failed to create vault at {vault_path}")
    return vault_path

def resolve_repo_path(path):
    """Resolve relative paths against the repository root"""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p

def load_vault_content(vault_path):
    """Load vault content from file"""
    vault_path = ensure_vault_exists(vault_path)
    if not vault_path.exists():
        return []
    with open(vault_path, "r", encoding='utf-8') as vault_file:
        return vault_file.readlines()

def load_or_generate_embeddings(vault_content, embeddings_file, embedding_model='mxbai-embed-large'):
    """Load embeddings from cache or generate new ones"""
    embeddings_path = resolve_repo_path(embeddings_file)
    if embeddings_path.exists():
        print(f"Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'r') as f:
            cached_embeddings = json.load(f)
        if len(cached_embeddings) == len(vault_content):
            return torch.tensor(cached_embeddings)
        else:
            print("Embeddings cache is outdated, regenerating...")

    print("Generating embeddings for vault content...")
    vault_embeddings = []
    for i, content in enumerate(vault_content):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(vault_content)}")
        response = ollama.embeddings(model=embedding_model, prompt=content)
        vault_embeddings.append(response["embedding"])

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(embeddings_path, 'w') as f:
        json.dump(vault_embeddings, f)
    print(f"Saved embeddings to {embeddings_path}")

    return torch.tensor(vault_embeddings)

def build_bm25_index(vault_content):
    """Build BM25 index from vault content"""
    tokenized_docs = [doc.lower().split() for doc in vault_content]
    return BM25Okapi(tokenized_docs)

def get_hybrid_context(query, vault_embeddings, vault_content, bm25_index, 
                       top_k=3, weight_vector=0.6, weight_bm25=0.4):
    """Retrieve context using hybrid retrieval (vector + BM25 keyword search)"""
    if vault_embeddings.nelement() == 0:
        return []

    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=query)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    vector_scores = cos_scores.numpy()
    
    vector_scores_min = vector_scores.min()
    vector_scores_max = vector_scores.max()
    if vector_scores_max - vector_scores_min > 1e-10:
        vector_scores_normalized = (vector_scores - vector_scores_min) / (vector_scores_max - vector_scores_min)
    else:
        vector_scores_normalized = np.zeros_like(vector_scores)
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    bm25_scores_min = bm25_scores.min()
    bm25_scores_max = bm25_scores.max()
    if bm25_scores_max - bm25_scores_min > 1e-10:
        bm25_scores_normalized = (bm25_scores - bm25_scores_min) / (bm25_scores_max - bm25_scores_min)
    else:
        bm25_scores_normalized = np.zeros_like(bm25_scores)
    
    combined_scores = (weight_vector * vector_scores_normalized + 
                      weight_bm25 * bm25_scores_normalized)
    
    top_k = min(top_k, len(combined_scores))
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def generate_response(query, context, system_message, ollama_model, client):
    """Generate response using LLM with context"""
    user_input_with_context = query
    if context:
        context_str = "\n".join(context)
        user_input_with_context = query + "\n\nRelevant Context:\n" + context_str

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input_with_context}
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )

    return response.choices[0].message.content

class HybridRAGPipeline:
    """Hybrid Retrieval-Augmented Generation pipeline combining vector embeddings + BM25 keyword search"""

    def __init__(self, config, weight_vector=0.6, weight_bm25=0.4):
        """
        Initialize hybrid RAG pipeline.
        
        Args:
            config: Configuration dictionary loaded from config.yaml
            weight_vector: Weight for semantic similarity scores (default: 0.6)
            weight_bm25: Weight for BM25 keyword scores (default: 0.4)
        """
        self.config = config
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25
        self.client = OpenAI(
            base_url=config['ollama_api']['base_url'],
            api_key=config['ollama_api']['api_key']
        )
        self.vault_content = load_vault_content(config['vault_file'])
        self.vault_embeddings = load_or_generate_embeddings(
            self.vault_content,
            config['embeddings_file'],
            'mxbai-embed-large'
        )
        print("Building BM25 index for keyword search...")
        self.bm25_index = build_bm25_index(self.vault_content)
        print("BM25 index built successfully!")

    def process_query(self, query):
        """Process a query through the hybrid RAG pipeline"""
        context = get_hybrid_context(
            query, 
            self.vault_embeddings, 
            self.vault_content, 
            self.bm25_index,
            self.config['top_k'],
            weight_vector=self.weight_vector,
            weight_bm25=self.weight_bm25
        )
        
        if context:
            context_str = "\n".join(context)
            print("Context Pulled from Documents (Hybrid Retrieval): \n\n" + CYAN + context_str + RESET_COLOR)
        else:
            print(CYAN + "No relevant context found." + RESET_COLOR)

        response = generate_response(query, context, self.config['system_message'], self.config['ollama_model'], self.client)

        return {
            'answer': response,
            'context': context
        }

def main():
    """Main function for standalone hybrid RAG pipeline execution"""
    config = load_config()
    pipeline = HybridRAGPipeline(config, weight_vector=0.6, weight_bm25=0.4)

    print("Starting Hybrid RAG conversation loop...")
    while True:
        user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
        if user_input.lower() == 'quit':
            break

        result = pipeline.process_query(user_input)
        print(NEON_GREEN + "Response: \n\n" + result['answer'] + RESET_COLOR)

if __name__ == "__main__":
    main()
