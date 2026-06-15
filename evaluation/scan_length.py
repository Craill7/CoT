import pandas as pd
import glob
import os
import argparse
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="全量扫描 Parquet 数据集的 Token 长度")
    parser.add_argument("--data_dir", type=str, 
                        default="/mdr5/user/quantaalpha/zhouyan/cyh/datasets/cleaned_data/OpenR1-Math-220k",
                        help="包含 Parquet 文件的目录")
    # 👇 这里已经修改为正确的绝世无双 A800 本地路径
    parser.add_argument("--model_path", type=str, 
                        default="/nfsdata-117/model/Qwen2.5-32B-Instruct",
                        help="用来真实模拟分词的模型路径")
    parser.add_argument("--col", type=str, default="problem", 
                        help="要扫描长度的列名（默认 problem）")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载 Tokenizer
    print(f"🔄 正在加载 Tokenizer: {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. 获取文件
    file_pattern = os.path.join(args.data_dir, "train-000*-of-00010.parquet")
    files = glob.glob(file_pattern)
    files.sort()
    
    if not files:
        print("❌ 未找到 Parquet 文件，请检查路径！")
        return

    print(f"📦 共找到 {len(files)} 个文件，开始逐个扫描...")

    all_lengths = []
    max_len = 0
    max_text = ""
    max_file = ""

    # 3. 逐个文件处理，防内存溢出
    for file_path in files:
        file_name = os.path.basename(file_path)
        df = pd.read_parquet(file_path)
        
        if args.col not in df.columns:
            print(f"⚠️ 警告: 文件 {file_name} 中没有找到 '{args.col}' 列！")
            continue
            
        texts = df[args.col].astype(str).tolist()
        
        # 为了还原最真实的输入，加上 Prompt 模板的固定长度
        template_prefix = "Problem:\n"
        template_suffix = "\nPlease reason step by step and put your final answer within \\boxed{}."
        
        for text in tqdm(texts, desc=f"Scanning {file_name}"):
            full_prompt = template_prefix + text + template_suffix
            # 真实编码算 Token 数
            token_len = len(tokenizer.encode(full_prompt))
            all_lengths.append(token_len)
            
            if token_len > max_len:
                max_len = token_len
                max_text = full_prompt
                max_file = file_name

    # 4. 统计与分析
    if not all_lengths:
        print("❌ 没有读取到任何文本，退出。")
        return

    lengths_arr = np.array(all_lengths)
    
    print("\n" + "="*50)
    print("📊 数据集 Token 长度扫描报告")
    print("="*50)
    print(f"总计扫描条数: {len(lengths_arr)}")
    print(f"平均 Token 长度: {np.mean(lengths_arr):.2f}")
    print(f"中位数 (P50): {np.percentile(lengths_arr, 50):.0f}")
    print(f"P90 长度: {np.percentile(lengths_arr, 90):.0f}")
    print(f"P95 长度: {np.percentile(lengths_arr, 95):.0f}  <-- (截断推荐)")
    print(f"P99 长度: {np.percentile(lengths_arr, 99):.0f}  <-- (max_model_len 参考)")
    print(f"🔥 最大 Token 长度 (Max): {max_len}")
    print("="*50)
    print(f"🚨 最长的那条数据出现在文件 [{max_file}] 中，预览前 200 个字符：\n{max_text[:200]}...")

if __name__ == "__main__":
    main()