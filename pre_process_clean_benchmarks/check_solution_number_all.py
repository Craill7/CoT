import json
from collections import Counter
from glob import glob

# ================= 配置区 =================
DATA_DIR = "/mdr0/user/quantaalpha/zhouyan/cyh/datasets/raw_data/OpenR1-Math-220k/data"
# ==========================================

files = sorted(glob(f"{DATA_DIR}/train-*.parquet"))
print(f"📂 找到 {len(files)} 个 parquet 文件\n")

total_rows = 0
total_cots = 0
counts = []
nan_count = 0
empty_count = 0

for f in files:
    import pandas as pd
    fname = f.split("/")[-1]
    df = pd.read_parquet(f)
    total_rows += len(df)
    
    for _, row in df.iterrows():
        gens = row['generations']
        
        # 【关键修复】统一转为 list
        if type(gens).__name__ == 'ndarray':
            gens = gens.tolist()
        elif isinstance(gens, str):
            try:
                gens = json.loads(gens)
            except:
                nan_count += 1
                counts.append(0)
                continue
        elif not isinstance(gens, list):
            nan_count += 1
            counts.append(0)
            continue
        
        # 过滤空字符串，统计有效 CoT 数
        valid = 0
        for g in gens:
            if g and isinstance(g, str) and g.strip():
                valid += 1
        
        if valid == 0:
            empty_count += 1
        counts.append(valid)
        total_cots += valid
    
    print(f"  ✅ {fname}: {len(df)} 条")

counter = Counter(counts)
print(f"\n{'='*50}")
print(f"📊 源头数据统计报告")
print(f"{'='*50}")
print(f"总题目数:         {total_rows}")
print(f"总 CoT 数:        {total_cots}")
print(f"平均每题 CoT 数:  {total_cots / total_rows:.2f}")
print(f"generations缺失:  {nan_count} 条")
print(f"generations全空:  {empty_count} 条")
print(f"\n📋 每题 CoT 数量分布:")
for n in sorted(counter.keys()):
    cnt = counter[n]
    pct = cnt / total_rows * 100
    bar = "█" * int(pct)
    print(f"  {n} 条 CoT: {cnt:>7} 题 ({pct:>5.1f}%) {bar}")
