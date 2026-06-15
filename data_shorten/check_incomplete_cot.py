import os
import glob
import pandas as pd
from tqdm import tqdm

def main():
    # 数据集所在的目录和文件匹配模式
    base_dir = "/mdr0/user/quantaalpha/zhouyan/cyh/datasets/cleaned_data/OpenR1-Math-220k"
    file_pattern = os.path.join(base_dir, "train-0000*-of-00010.parquet")
    
    # 获取所有的 parquet 文件 (应为 10 个)
    parquet_files = sorted(glob.glob(file_pattern))
    
    if not parquet_files:
        print(f"❌ 未找到任何匹配的 Parquet 文件，请检查路径: {file_pattern}")
        return

    print(f"📂 发现 {len(parquet_files)} 个 Parquet 文件，准备扫描...")

    total_questions = 0
    total_cots = 0
    incomplete_cots = 0
    questions_affected = 0

    # 遍历每个文件
    for file_path in tqdm(parquet_files, desc="🔍 扫描数据分布"):
        try:
            # 读取 Parquet 文件
            df = pd.read_parquet(file_path, columns=['is_reasoning_complete'])
            total_questions += len(df)
            
            # 遍历该列的每一行 (每一行是一个布尔列表，如 [True, False])
            for flags in df['is_reasoning_complete']:
                if flags is None:
                    continue
                
                # 累加这一题产生的总 CoT 数量
                total_cots += len(flags)
                
                # 统计这题里有多少个 False (即未完整生成的 CoT)
                # parquet 读取出来的可能是 numpy array 或 list
                incomp_count = sum(1 for flag in flags if not flag)
                
                incomplete_cots += incomp_count
                if incomp_count > 0:
                    questions_affected += 1
                    
        except Exception as e:
            print(f"\n⚠️ 读取文件 {file_path} 时出错: {e}")

    # --- 打印统计报告 ---
    print("\n" + "="*50)
    print("📈 OpenR1-Math-220k 数据完整性体检报告")
    print("-" * 50)
    print(f"🔸 扫描题目总数: {total_questions:,} 题")
    print(f"🔸 扫描 CoT 总数:  {total_cots:,} 条")
    print("-" * 50)
    print(f"✅ 完整闭环的 CoT: {total_cots - incomplete_cots:,} 条")
    print(f"❌ 未完整生成的 CoT: {incomplete_cots:,} 条")
    print("-" * 50)
    
    # 计算占比
    if total_cots > 0:
        ratio = (incomplete_cots / total_cots) * 100
        print(f"⚠️ 残次 CoT 占比: {ratio:.4f}%")
        
        q_ratio = (questions_affected / total_questions) * 100
        print(f"⚠️ 包含残次 CoT 的题目占比: {q_ratio:.2f}% ({questions_affected:,} 题)")
    else:
        print("未统计到任何 CoT 数据。")
    print("="*50)

if __name__ == "__main__":
    main()