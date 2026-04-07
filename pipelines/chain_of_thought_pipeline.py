import yaml
from openai import OpenAI
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

def generate_chain_of_thought(query, system_message, ollama_model, client):
    """Generate response using Chain of Thought reasoning"""
    prompt = f"""Think through this question step by step. Show your reasoning clearly.

    Question: {query}

    IMPORTANT: After showing your reasoning, write "Final Answer:" followed by ONLY the answer itself (no explanation). For yes/no questions, write only "yes" or "no"."""

    messages = [
        {"role": "system", "content": system_message},
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

    # Extract answer if common separators are found
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

class ChainOfThoughtPipeline:
    """Chain of Thought Pipeline - standalone reasoning without external knowledge"""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config['ollama_api']['base_url'],
            api_key=config['ollama_api']['api_key']
        )
        self.system_message = "You are a helpful assistant. Think step by step through your reasoning, then provide ONLY the final answer. For yes/no questions, answer with only 'yes' or 'no'. Do not include any explanation after your final answer."

    def process_query(self, query):
        """Process a query through the Chain of Thought pipeline"""
        print(PINK + f"Processing query with Chain of Thought: {query}" + RESET_COLOR)

        reasoning, answer = generate_chain_of_thought(
            query,
            self.system_message,
            self.config['ollama_model'],
            self.client
        )

        print(CYAN + f"Reasoning: {reasoning}" + RESET_COLOR)
        print(NEON_GREEN + f"Final Answer: {answer}" + RESET_COLOR)

        return {'answer': answer, 'reasoning': reasoning, 'context': []}

def main():
    """Main function for standalone CoT pipeline execution"""
    config = load_config()
    pipeline = ChainOfThoughtPipeline(config)

    print("Starting Chain of Thought conversation loop...")
    while True:
        user_input = input(YELLOW + "Ask a question (or type 'quit' to exit): " + RESET_COLOR)
        if user_input.lower() == 'quit':
            break

        result = pipeline.process_query(user_input)
        print(NEON_GREEN + "Response: \n\n" + result['answer'] + RESET_COLOR)

if __name__ == "__main__":
    main()