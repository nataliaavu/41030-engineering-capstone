import json
import os
from collections import Counter

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

    levels = [item.get('level') for item in data if 'level' in item]
    if levels:
        level_counts = Counter(levels)
        print(f"Dataset level counts (sample): {level_counts.most_common(10)}")

    if any('level' in item for item in data):
        subset = [item for item in data if str(item.get('level', '')).lower() == str(level).lower()][:n]
        if not subset:
            print(f"No items found with level '{level}' (case-insensitive). Falling back to first {n} items.")
            subset = data[:n]
    else:
        subset = data[:n]

    with open(subset_path, 'w', encoding='utf-8') as sf:
        json.dump(subset, sf, ensure_ascii=False, indent=2)
    print(f"Wrote subset ({len(subset)}) to {subset_path}")

    written = 0
    with open(vault_path, 'w', encoding='utf-8') as vault:
        for item in subset:
            supporting = item.get('supporting_facts', [])
            context = item.get('context', [])
            for pair in supporting:
                if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
                    print(f"Skipping unexpected supporting_facts entry: {pair}")
                    continue
                title, sent_id = pair[0], pair[1]
                for ctx_title, sentences in context:
                    if ctx_title == title:
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

if __name__ == '__main__':
    create_vault_from_hotpot()
