import json
import yaml
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.rag_pipeline import RAGPipeline, load_config as load_rag_config
from pipelines.chain_of_thought_pipeline import ChainOfThoughtPipeline
from pipelines.baseline_pipeline import BaselinePipeline

def load_evaluation_data(data_path='data/hotpot_subset.json'):
    """Load evaluation data from JSON file"""
    if not os.path.isabs(data_path):
        data_path = REPO_ROOT / data_path
    if not os.path.exists(data_path):
        print(f"Evaluation data not found at: {data_path}")
        return []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract questions and answers
    eval_data = []
    for item in data:
        question = item.get('question', '')
        answer = item.get('answer', '')
        if question and answer:
            eval_data.append({
                'question': question,
                'answer': answer,
                'context': item.get('context', [])
            })

    return eval_data

def evaluate_pipeline(pipeline, eval_data, max_samples=10):
    """Evaluate a pipeline on test data"""
    results = []
    correct = 0
    total = min(len(eval_data), max_samples)

    print(f"Evaluating {pipeline.__class__.__name__} on {total} samples...")

    for i, item in enumerate(eval_data[:max_samples]):
        print(f"\n--- Sample {i+1}/{total} ---")
        question = item['question']
        expected_answer = item['answer']

        try:
            result = pipeline.process_query(question)
            predicted_answer = result['answer']

            # Simple exact match evaluation (can be improved)
            is_correct = expected_answer.lower().strip() in predicted_answer.lower()

            results.append({
                'question': question,
                'expected': expected_answer,
                'predicted': predicted_answer,
                'correct': is_correct,
                'reasoning': result.get('reasoning'),
                'context_used': len(result.get('context', [])) > 0
            })

            if is_correct:
                correct += 1

            print(f"Expected: {expected_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {is_correct}")

        except Exception as e:
            print(f"Error processing question: {e}")
            results.append({
                'question': question,
                'expected': expected_answer,
                'predicted': '',
                'correct': False,
                'error': str(e)
            })

    accuracy = correct / total if total > 0 else 0
    return results, accuracy

def run_comparison(eval_data, config_path=None, max_samples=5):
    """Run all pipelines and compare results"""
    config_path = config_path or REPO_ROOT / 'config.yaml'
    config = load_rag_config(config_path)

    # Initialize pipelines
    pipelines = {
        'RAG': RAGPipeline(config),
        'Chain_of_Thought': ChainOfThoughtPipeline(config),
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

    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")

    for name, acc in accuracies.items():
        print(f"{name}: {acc:.2%}")

    # Save detailed results
    with open(REPO_ROOT / 'evaluation_results.json', 'w') as f:
        json.dump({
            'accuracies': accuracies,
            'results': results
        }, f, indent=2)

    print("\nDetailed results saved to evaluation_results.json")

    return results, accuracies

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipelines")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data", default="data/hotpot_subset.json", help="Evaluation data path")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--pipeline", choices=['rag', 'cot', 'baseline', 'all'], default='all',
                       help="Which pipeline to evaluate")

    args = parser.parse_args()

    # Load evaluation data
    eval_data = load_evaluation_data(args.data)
    if not eval_data:
        print("No evaluation data found. Please check the data path.")
        return

    config_path = Path(args.config) if os.path.isabs(args.config) else REPO_ROOT / args.config
    config = load_rag_config(config_path)

    if args.pipeline == 'all':
        run_comparison(eval_data, config_path, args.samples)
    else:
        # Run single pipeline
        pipeline_map = {
            'rag': RAGPipeline,
            'cot': ChainOfThoughtPipeline,
            'baseline': BaselinePipeline
        }

        pipeline_class = pipeline_map[args.pipeline]
        pipeline = pipeline_class(config)
        results, accuracy = evaluate_pipeline(pipeline, eval_data, args.samples)

        print(f"\n{args.pipeline.upper()} Pipeline Accuracy: {accuracy:.2%}")

        # Save results
        with open(f'{args.pipeline}_results.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()