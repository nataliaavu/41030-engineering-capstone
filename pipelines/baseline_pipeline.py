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

def generate_direct_response(query, system_message, ollama_model, client, max_tokens, temperature):
    """Generate direct response without any augmentation"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content

class BaselinePipeline:
    """Baseline Pipeline - direct LLM generation without RAG or CoT"""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config['ollama_api']['base_url'],
            api_key=config['ollama_api']['api_key']
        )
        self.system_message = "You are a helpful assistant. Answer the question with ONLY a short, direct answer. For yes/no questions, answer with only 'yes' or 'no'. Do not include explanations."

    def process_query(self, query):
        """Process a query through the baseline pipeline"""
        print(PINK + f"Processing query directly: {query}" + RESET_COLOR)

        answer = generate_direct_response(
            query,
            self.system_message,
            self.config['ollama_model'],
            self.client,
            self.config.get('max_tokens', 2000),
            self.config.get('temperature', 0.1)
        )

        print(NEON_GREEN + f"Direct Answer: {answer}" + RESET_COLOR)

        return {
            'answer': answer,
            'reasoning': None,
            'context': []
        }

def main():
    """Main function for standalone baseline pipeline execution"""
    config = load_config()
    pipeline = BaselinePipeline(config)

    print("Starting Baseline conversation loop...")
    while True:
        user_input = input(YELLOW + "Ask a question (or type 'quit' to exit): " + RESET_COLOR)
        if user_input.lower() == 'quit':
            break

        result = pipeline.process_query(user_input)
        print(NEON_GREEN + "Response: \n\n" + result['answer'] + RESET_COLOR)

if __name__ == "__main__":
    main()