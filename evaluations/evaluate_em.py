import json
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.rag_pipeline import RAGPipeline, load_config
from pipelines.chain_of_thought_pipeline import ChainOfThoughtPipeline
from pipelines.baseline_pipeline import BaselinePipeline
from pipelines.cot_and_rag_pipeline import COTRAGPipeline

def normalize_answer(text):
    """Normalize answer for exact match comparison following standard QA metric practice"""
    import re
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_evaluation_data(data_path='data/hotpot_subset.json'):
    """Load evaluation data from JSON file"""
    data_path = Path(data_path) if Path(data_path).is_absolute() else REPO_ROOT / data_path
    if not data_path.exists():
        print(f"Evaluation data not found at: {data_path}")
        return []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [
        {
            'question': item['question'],
            'answer': item['answer'],
            'context': item.get('context', [])
        }
        for item in data
        if item.get('question') and item.get('answer')
    ]

def evaluate_pipeline(pipeline, eval_data, max_samples=300):
    """Evaluate a pipeline on test data"""
    results = []
    correct = 0
    total = min(len(eval_data), max_samples)

    print(f"Evaluating {pipeline.__class__.__name__} on {total} samples...")

    for i, item in enumerate(eval_data[:max_samples]):
        print(f"\n--- Sample {i+1}/{total} ---")
        question = item['question']
        expected_answer = normalize_answer(item['answer'])

        try:
            result = pipeline.process_query(question)
            predicted_answer = normalize_answer(result['answer'])
            is_correct = expected_answer.lower().strip() in predicted_answer.lower()

            results.append({
                'question': question,
                'expected': item['answer'],
                'predicted': result['answer'],
                'correct': is_correct,
                'reasoning': result.get('reasoning'),
                'context_used': len(result.get('context', [])) > 0
            })

            if is_correct:
                correct += 1

            print(f"Expected: {item['answer']}")
            print(f"Predicted: {result['answer']}")
            print(f"Correct: {is_correct}")

        except Exception as e:
            print(f"Error processing question: {e}")
            results.append({
                'question': question,
                'expected': item['answer'],
                'predicted': '',
                'correct': False,
                'error': str(e)
            })

    accuracy = correct / total if total > 0 else 0
    return results, accuracy

def run_comparison(eval_data, config_path, max_samples=5):
    """Run all pipelines and compare results"""
    config = load_config(config_path)
    pipelines = {
        'RAG': RAGPipeline(config),
        'Chain_of_Thought': ChainOfThoughtPipeline(config),
        'COT_RAG': COTRAGPipeline(config),
        'Baseline': BaselinePipeline(config)
    }

    results = {}
    accuracies = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name} Pipeline")
        print(f"{'='*50}")
        pipeline_results, accuracy = evaluate_pipeline(pipeline, eval_data, max_samples)
        results[name] = pipeline_results
        accuracies[name] = accuracy

    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    for name, acc in accuracies.items():
        print(f"{name}: {acc:.2%}")

    output_file = REPO_ROOT / 'evaluations' / 'em_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({'accuracies': accuracies, 'results': results}, f, indent=2)
    print(f"\nDetailed results saved to {output_file.name}")

    return results, accuracies

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA pipelines")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / 'config.yaml', help="Config file path")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / 'data' / 'hotpot_subset.json', help="Evaluation data path")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--pipeline", choices=['rag', 'cot', 'baseline', 'all'], default='all', help="Which pipeline to evaluate")
    args = parser.parse_args()

    eval_data = load_evaluation_data(str(args.data))
    if not eval_data:
        print("No evaluation data found.")
        return

    config_path = args.config
    config = load_config(config_path)

    if args.pipeline == 'all':
        run_comparison(eval_data, config_path, args.samples)
    else:
        pipeline_map = {'rag': RAGPipeline, 'cot': ChainOfThoughtPipeline, 'baseline': BaselinePipeline, 'cot_rag': COTRAGPipeline}
        pipeline = pipeline_map[args.pipeline](config)
        results, accuracy = evaluate_pipeline(pipeline, eval_data, args.samples)
        print(f"\n{args.pipeline.upper()} Pipeline Accuracy: {accuracy:.2%}")
        with open(f'{args.pipeline}_results.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()