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
    prompt = f"""Please think through this question step by step, showing your reasoning process clearly.

Question: {query}

Please provide your final answer after showing your step-by-step reasoning."""

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

    # Try to separate reasoning from final answer
    # Look for common patterns
    reasoning = full_response
    answer = full_response

    # Common separators
    separators = ["Final Answer:", "Answer:", "Conclusion:"]
    for sep in separators:
        if sep in full_response:
            parts = full_response.split(sep, 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
            break

    return reasoning, answer

class ChainOfThoughtPipeline:
    """Chain of Thought Pipeline - standalone reasoning without external knowledge"""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config['ollama_api']['base_url'],
            api_key=config['ollama_api']['api_key']
        )
        self.system_message = "You are a helpful assistant that thinks step by step to provide accurate answers. Always show your reasoning process clearly before giving the final answer."

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

        return {
            'answer': answer,
            'reasoning': reasoning,
            'context': []  # No external context used
        }

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