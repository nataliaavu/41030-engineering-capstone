import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from openai import OpenAI
import yaml
from .rag_pipeline import (
    get_relevant_context,
    load_or_generate_embeddings,
    load_config,
    load_vault_content,
)

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def generate_cot_rag_response(query, context, system_message, ollama_model, client):
    """Generate response using LLM with context and Chain of Thought reasoning"""
    user_input_with_context = query
    if context:
        context_str = "\n".join(context)
        user_input_with_context = query + "\n\nRelevant Context:\n" + context_str

    prompt = f"""Think through this question step by step. Show your reasoning clearly.

    Question: {query}

    IMPORTANT: After showing your reasoning, write "Final Answer:" followed by ONLY the answer itself (no explanation). For yes/no questions, write only "yes" or "no"."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input_with_context},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
        temperature=0.1,
    )

    full_response = response.choices[0].message.content
    reasoning = full_response
    answer = full_response

    separators = ["Final Answer:", "final answer:", "Therefore, my final answer is:", "Conclusion:", "conclusion:", "Answer:", "answer is:"]
    for sep in separators:
        if sep in full_response:
            parts = full_response.split(sep, 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else answer
            answer = answer.split('\n')[0].strip()
            break
    
    if answer == full_response:
        lines = [line.strip() for line in full_response.split('\n') if line.strip()]
        if lines:
            answer = lines[-1]
            for prefix in ["**No**", "**Yes**", "**", "Final answer:", "Answer:"]:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()

    return reasoning, answer

class COTRAGPipeline:
    """Chain of Thought + Retrieval-Augmented Generation pipeline"""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config['ollama_api']['base_url'],
            api_key=config['ollama_api']['api_key']
        )
        self.system_message = "You are a helpful assistant. Think step by step through your reasoning, then provide ONLY the final answer. For yes/no questions, answer with only 'yes' or 'no'. Do not include any explanation after your final answer."
        self.vault_content = load_vault_content(config['vault_file'])
        self.vault_embeddings = load_or_generate_embeddings(
            self.vault_content,
            config['embeddings_file'],
            'mxbai-embed-large'
        )

    def process_query(self, query):
        """Process a query through the RAG pipeline"""

        context = get_relevant_context(query, self.vault_embeddings, self.vault_content, self.config['top_k'])
        if context:
            context_str = "\n".join(context)
            print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
        else:
            print(CYAN + "No relevant context found." + RESET_COLOR)

        reasoning, response = generate_cot_rag_response(
            query, 
            context, 
            self.system_message, 
            self.config['ollama_model'], 
            self.client
            )
        
        print(CYAN + f"Reasoning: {reasoning}" + RESET_COLOR)
        print(NEON_GREEN + f"Final Answer: {response}" + RESET_COLOR)


        return {
            'answer': response,
            'reasoning': reasoning, 
            'context': context
        }

def main():
    """Main function for combined COT and RAG pipeline execution"""
    config = load_config()
    pipeline = COTRAGPipeline(config)

    print("Starting COT + RAG conversation loop...")
    while True:
        user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
        if user_input.lower() == 'quit':
            break

        result = pipeline.process_query(user_input)
        print(NEON_GREEN + "Response: \n\n" + result['answer'] + RESET_COLOR)

if __name__ == "__main__":
    main()