import json

from collections import Counter

counts = []
with open("/home/zhouyan/share/cyh/CoT/sample_data/train_sampled_3k.jsonl", 'r') as f:
    for line in f:
        item = json.loads(line)
        n = len(item.get('generations', []))
        counts.append(n)

counter = Counter(counts)
for n, cnt in sorted(counter.items()):
    print(f"有 {n} 条 CoT 的题目: {cnt} 个")

print(f"\n总 CoT 数: {sum(counts)}")
