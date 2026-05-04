import json
import argparse
import re
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.rag_pipeline import RAGPipeline, load_config
from pipelines.chain_of_thought_pipeline import ChainOfThoughtPipeline
from pipelines.baseline_pipeline import BaselinePipeline
from pipelines.cot_and_rag_pipeline import COTRAGPipeline

def compute_f1(predicted, expected):
    """
    Compute F1 score between predicted and expected answers using token frequency overlap.
    This follows the standard SQuAD F1 score calculation which counts token occurrences,
    not just unique tokens. This is more reliable than set-based overlap for QA evaluation.
    """
    # Tokenize and lowercase
    pred_tokens = re.findall(r'\b\w+\b', predicted.lower())
    exp_tokens = re.findall(r'\b\w+\b', expected.lower())

    # Handle empty cases
    if not pred_tokens and not exp_tokens:
        return 1.0  # Both empty, perfect match
    if not pred_tokens or not exp_tokens:
        return 0.0  # One empty, no overlap

    # Count token frequencies (not just unique tokens)
    pred_counter = Counter(pred_tokens)
    exp_counter = Counter(exp_tokens)

    # Calculate intersection based on token counts
    # For each token, take the minimum count between predicted and expected
    common_tokens = 0
    for token in pred_counter:
        if token in exp_counter:
            common_tokens += min(pred_counter[token], exp_counter[token])

    # Calculate precision and recall
    precision = common_tokens / len(pred_tokens) if pred_tokens else 0.0
    recall = common_tokens / len(exp_tokens) if exp_tokens else 0.0
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

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

def evaluate_pipeline(pipeline, eval_data, max_samples=1000):
    """Evaluate a pipeline on test data using F1 scoring"""
    results = []
    total_f1 = 0
    total = min(len(eval_data), max_samples)

    print(f"Evaluating {pipeline.__class__.__name__} on {total} samples...")

    for i, item in enumerate(eval_data[:max_samples]):
        print(f"\n--- Sample {i+1}/{total} ---")
        question = item['question']
        expected_answer = item['answer']

        try:
            result = pipeline.process_query(question)
            predicted_answer = result['answer']
            f1_score = compute_f1(predicted_answer, expected_answer)

            results.append({
                'question': question,
                'expected': expected_answer,
                'predicted': predicted_answer,
                'f1_score': f1_score,
                'reasoning': result.get('reasoning'),
                'context_used': len(result.get('context', [])) > 0
            })

            total_f1 += f1_score
            print(f"Expected: {expected_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"F1 Score: {f1_score:.3f}")

        except Exception as e:
            print(f"Error processing question: {e}")
            results.append({
                'question': question,
                'expected': expected_answer,
                'predicted': '',
                'f1_score': 0.0,
                'error': str(e)
            })

    average_f1 = total_f1 / total if total > 0 else 0
    return results, average_f1

def run_comparison(eval_data, config_path, max_samples=5):
    """Run all pipelines and compare F1 results"""
    config = load_config(config_path)
    pipelines = {
        'RAG': RAGPipeline(config),
        'Chain_of_Thought': ChainOfThoughtPipeline(config),
        'COT_RAG': COTRAGPipeline(config),
        'Baseline': BaselinePipeline(config)
    }

    results = {}
    f1_scores = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name} Pipeline")
        print(f"{'='*50}")
        pipeline_results, average_f1 = evaluate_pipeline(pipeline, eval_data, max_samples)
        results[name] = pipeline_results
        f1_scores[name] = average_f1

    # Print summary
    print(f"\n{'='*50}")
    print("F1 SCORE SUMMARY")
    print(f"{'='*50}")
    for name, f1 in f1_scores.items():
        print(f"{name}: {f1:.3f}")

    # Save detailed results
    output_file = REPO_ROOT / 'evaluations' / 'f1_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({'f1_scores': f1_scores, 'results': results}, f, indent=2)
    print(f"\nDetailed results saved to {output_file.name}")

    return results, f1_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA pipelines with F1 scoring")
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
        results, average_f1 = evaluate_pipeline(pipeline, eval_data, args.samples)
        print(f"\n{args.pipeline.upper()} Pipeline F1: {average_f1:.3f}")
        with open(f'{args.pipeline}_f1_results.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()