import pandas as pd
import glob
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="从分块 Parquet 文件中随机抽取数据")
    
    # 1. 输入目录：默认是你给的那个路径
    parser.add_argument("--input_dir", type=str, 
                        default="/mdr5/user/quantaalpha/zhouyan/cyh/datasets/cleaned_data/OpenR1-Math-220k",
                        help="包含 Parquet 文件的原始数据目录")
    
    # 2. 输出路径：默认在当前目录下生成
    parser.add_argument("--output_path", type=str, 
                        default="/home/zhouyan/share/cyh/CoT/sample_data/train_sampled_3k",
                        help="输出文件的基本路径和名称（不含后缀）")
    
    # 3. 采样数量：默认 3000
    parser.add_argument("--n", type=int, default=3000, help="随机抽取的条数")
    
    # 4. 随机种子：固定种子保证结果可复现
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 构造匹配模式：匹配 00000 到 00009
    file_pattern = os.path.join(args.input_dir, "train-0000*-of-00010.parquet")
    
    # 获取文件列表
    all_files = glob.glob(file_pattern)
    all_files.sort()
    
    if not all_files:
        print(f"❌ 错误：在 {args.input_dir} 下没找到匹配 train-0000*-of-00010.parquet 的文件")
        return

    print(f"📦 正在从 {len(all_files)} 个文件中读取数据...")
    
    # 读取并合并
    df_list = [pd.read_parquet(f) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    print(f"📊 全量数据加载完毕，共 {len(full_df)} 条。")

    # 检查采样数是否合理
    sample_n = min(args.n, len(full_df))
    if sample_n < args.n:
        print(f"⚠️ 警告：总数据量不足 {args.n} 条，将提取全部数据。")

    # 执行采样
    sampled_df = full_df.sample(n=sample_n, random_state=args.seed)

    # 导出文件
    # 自动处理输出目录是否存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 已创建输出目录: {output_dir}")

    # 保存为 Parquet
    parquet_out = f"{args.output_path}.parquet"
    sampled_df.to_parquet(parquet_out, index=False)
    
    # 保存为 JSONL
    jsonl_out = f"{args.output_path}.jsonl"
    sampled_df.to_json(jsonl_out, orient="records", lines=True, force_ascii=False)

    print(f"\n🎉 抽样任务完成！")
    print(f"✅ Parquet 保存至: {parquet_out}")
    print(f"✅ JSONL 保存至: {jsonl_out}")
    print(f"🚀 总计抽取条数: {sample_n}")

if __name__ == "__main__":
    main()