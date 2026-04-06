import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def create_vault_from_hotpot(hotpot_path='data/hotpot_dev_distractor_v1.json',
                             subset_path='data/hotpot_subset.json',
                             vault_path='vault.txt',
                             level='easy',
                             n=5):
    if not os.path.exists(hotpot_path):
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


# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
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
    return json.dumps({"Rewritten Query": rewritten_query})
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Create the vault from the Hotpot dataset using the desired level (set to 'hard' for this dataset)
create_vault_from_hotpot(level='hard')

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])

# Convert to tensor and print embeddings
print("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# Conversation loop
print("Starting conversation loop...")
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Answer based only on the provided context."

while True:
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
