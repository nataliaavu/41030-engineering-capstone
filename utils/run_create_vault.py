import json
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def create_vault_from_hotpot(hotpot_path=None,
                             subset_path=None,
                             vault_path=None,
                             level='hard',
                             n=300):
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
        print(f"Dataset level counts (sample): {level_counts.most_common(300)}")

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

if __name__ == '__main__':
    create_vault_from_hotpot()
