import pandas as pd
import os

# 1. 具体文件路径（混合了 JSONL 和 Parquet）
benchmark_files = [
    # '/mdr5/user/quantaalpha/zhouyan/cyh/benchmarks/HuggingFaceH4_MATH-500/test.jsonl',
    # '/mdr5/user/quantaalpha/zhouyan/cyh/benchmarks/math-ai_amc23/test-00000-of-00001.parquet'
    # '/mdr5/user/quantaalpha/zhouyan/cyh/benchmarks/opencompass_AIME2025/aime2025.jsonl'
    '/home/zhouyan/share/cyh/CoT/3k_test_data/train_sampled_6k(question+cot).jsonl'
]

# 2. 采样结果保存根目录
output_root = '/home/zhouyan/share/cyh/CoT/sft'
os.makedirs(output_root, exist_ok=True)

print("🚀 开始并行采样，自动识别 JSONL/Parquet 格式...")

for f_path in benchmark_files:
    # --- 逻辑 A: 提取数据集名称 (保持你原有的逻辑，增加容错) ---
    path_parts = f_path.split('/')
    dataset_name = path_parts[-3] if 'data' in f_path else path_parts[-2]
    if dataset_name in ['data', 'test', 'test-00000-of-00001']:
        dataset_name = path_parts[-3] if len(path_parts) >= 3 else path_parts[-2]
    
    save_filename = os.path.join(output_root, f"{dataset_name}_sample.jsonl")
    
    try:
        # --- 逻辑 B: 根据后缀名动态选择读取器 ---
        if f_path.endswith('.parquet'):
            df = pd.read_parquet(f_path)
        elif f_path.endswith('.jsonl') or f_path.endswith('.json'):
            # lines=True 是读取 JSONL 的关键
            df = pd.read_json(f_path, lines=True)
        else:
            print(f"⚠️ 跳过未知格式: {f_path}")
            continue

        # --- 逻辑 C: 统一采样并保存 ---
        # 抽取前 5 条 (DataFrame 对象是统一的)
        df_mini = df.head(5)
        
        # 统一保存为 JSONL，方便后续查看
        df_mini.to_json(save_filename, orient='records', lines=True, force_ascii=False)
        
        print(f"✅ 已完成: {dataset_name}")
        print(f"   - 类型: {'Parquet' if f_path.endswith('.parquet') else 'JSONL'}")
        print(f"   - 路径: {save_filename}")
        print(f"   - 样本量: {len(df_mini)} | 总列名: {df.columns.tolist()}\n")
        
    except Exception as e:
        print(f"❌ 处理失败 {dataset_name}: {e}")

print(f"✨ 采样结束。请前往 {output_root} 查看。")