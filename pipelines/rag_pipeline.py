import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import ollama
from openai import OpenAI
import json
import yaml
from utils.run_create_vault import create_vault_from_hotpot

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def load_config(config_path=None):
    """Load configuration from YAML file"""
    config_path = config_path or ROOT_DIR / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_vault_exists(vault_path, level='hard', n=300):
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
        if i % 300 == 0:
            print(f"Processing {i}/{len(vault_content)}")
        response = ollama.embeddings(model=embedding_model, prompt=content)
        vault_embeddings.append(response["embedding"])

    # Save to cache
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(embeddings_path, 'w') as f:
        json.dump(vault_embeddings, f)
    print(f"Saved embeddings to {embeddings_path}")

    return torch.tensor(vault_embeddings)

def get_relevant_context(query, vault_embeddings, vault_content, top_k=3):
    """Get relevant context from vault based on query"""
    if vault_embeddings.nelement() == 0:
        return []

    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=query)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
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

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline"""

    def __init__(self, config):
        self.config = config
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

    def process_query(self, query):
        """Process a query through the RAG pipeline"""
        search_query = query

        # Retrieve relevant context
        context = get_relevant_context(search_query, self.vault_embeddings, self.vault_content, self.config['top_k'])
        if context:
            context_str = "\n".join(context)
            print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
        else:
            print(CYAN + "No relevant context found." + RESET_COLOR)

        # Generate response
        response = generate_response(query, context, self.config['system_message'], self.config['ollama_model'], self.client)

        return {
            'answer': response,
            'reasoning': None,  # RAG doesn't have explicit reasoning trace
            'context': context
        }

def main():
    """Main function for standalone RAG pipeline execution"""
    config = load_config()
    pipeline = RAGPipeline(config)

    print("Starting RAG conversation loop...")
    while True:
        user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
        if user_input.lower() == 'quit':
            break

        result = pipeline.process_query(user_input)
        print(NEON_GREEN + "Response: \n\n" + result['answer'] + RESET_COLOR)

if __name__ == "__main__":
    main()