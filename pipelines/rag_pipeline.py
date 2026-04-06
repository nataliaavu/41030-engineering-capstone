import torch
import ollama
import os
from openai import OpenAI
import json
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

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

def create_vault_from_hotpot(hotpot_path=None,
                             subset_path=None,
                             vault_path=None,
                             level='easy',
                             n=10):
    """Create vault from HotpotQA dataset"""
    hotpot_path = hotpot_path or ROOT_DIR / 'data' / 'hotpot_dev_distractor_v1.json'
    subset_path = subset_path or ROOT_DIR / 'data' / 'hotpot_subset.json'
    vault_path = vault_path or ROOT_DIR / 'vault.txt'
    hotpot_path = Path(hotpot_path)
    subset_path = Path(subset_path)
    vault_path = Path(vault_path)

    if not hotpot_path.exists():
        print(f"Hotpot file not found at: {hotpot_path}")
        return
    with open(hotpot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {hotpot_path}")

    # Report available level values (if any) and try a case-insensitive match
    from collections import Counter
    levels = [item.get('level') for item in data if 'level' in item]
    if levels:
        level_counts = Counter(levels)
        print(f"Dataset level counts (sample): {level_counts.most_common(10)}")

    # If dataset contains a 'level' key, do a case-insensitive match; if empty, stop (no fallback)
    if any('level' in item for item in data):
        subset = [item for item in data if str(item.get('level', '')).lower() == str(level).lower()][:n]
        if not subset:
            print(f"No items found with level '{level}' (case-insensitive). No subset will be created.")
            return
    else:
        subset = data[:n]

    # Save subset for inspection
    with open(subset_path, 'w', encoding='utf-8') as sf:
        json.dump(subset, sf, ensure_ascii=False, indent=2)
    print(f"Wrote subset ({len(subset)}) to {subset_path}")

    # Build vault robustly
    written = 0
    with open(vault_path, 'w', encoding='utf-8') as vault:
        for item in subset:
            supporting = item.get('supporting_facts', [])
            context = item.get('context', [])
            for pair in supporting:
                # support entries may be [title, sent_id]
                if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
                    print(f"Skipping unexpected supporting_facts entry: {pair}")
                    continue
                title, sent_id = pair[0], pair[1]
                # Find matching context title
                for ctx_title, sentences in context:
                    if ctx_title == title:
                        # guard index errors
                        try:
                            idx = int(sent_id)
                        except Exception:
                            print(f"Warning: non-integer sent_id {sent_id} for title {title}")
                            break
                        if isinstance(sentences, list) and 0 <= idx < len(sentences):
                            vault.write(sentences[idx].strip() + "\n")
                            written += 1
                        else:
                            print(f"Warning: bad sent_id {sent_id} for title {title}")
                        break
    print(f"Wrote {written} lines to {vault_path}")

def resolve_repo_path(path):
    """Resolve relative paths against the repository root"""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p

def load_vault_content(vault_path):
    """Load vault content from file"""
    vault_path = resolve_repo_path(vault_path)
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
        if i % 10 == 0:
            print(f"Processing {i}/{len(vault_content)}")
        response = ollama.embeddings(model=embedding_model, prompt=content)
        vault_embeddings.append(response["embedding"])

    # Save to cache
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

def rewrite_query(user_input, conversation_history, ollama_model, client):
    """Rewrite query based on conversation history"""
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:

    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

    Return ONLY the rewritten query text, without any additional formatting or explanations.

    Conversation History:
    {context}

    Original query: [{user_input}]

    Rewritten query:
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

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
    """RAG Pipeline with query rewriting"""

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
        self.conversation_history = []

    def process_query(self, query):
        """Process a query through the RAG pipeline"""
        # Rewrite query if not first turn
        if len(self.conversation_history) > 0:
            rewritten_query = rewrite_query(query, self.conversation_history, self.config['ollama_model'], self.client)
            print(PINK + f"Original Query: {query}" + RESET_COLOR)
            print(PINK + f"Rewritten Query: {rewritten_query}" + RESET_COLOR)
            search_query = rewritten_query
        else:
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

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})

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