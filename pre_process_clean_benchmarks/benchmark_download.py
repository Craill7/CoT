import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download



# 1. 精简后的核心数学 Benchmark 列表 (去重+评测必用)
math_datasets = [
    "math-ai/amc23"
]

# 2. 定义基础保存目录 (请确保该路径已创建或你有权限)
base_save_dir = "/mdr5/user/quantaalpha/zhouyan/cyh/benchmarks"

# 创建目录（如果不存在）
os.makedirs(base_save_dir, exist_ok=True)

print("🚀 开始批量拉取核心数学 Benchmark 数据集...")

for dataset_id in math_datasets:
    # 自动生成子目录名
    folder_name = dataset_id.replace("/", "_")
    local_dir = os.path.join(base_save_dir, folder_name)
    
    print(f"\n-----------------------------------------------")
    print(f"任务: {dataset_id}")
    print(f"目标路径: {local_dir}")
    print(f"-----------------------------------------------")
    
    try:
        snapshot_download(
            repo_id=dataset_id, 
            repo_type="dataset",  
            local_dir=local_dir,
            resume_download=True,   # 支持断点续传
            max_workers=4,          # 适当降低线程数防止被镜像站屏蔽，8容易断
            ignore_patterns=["*.pdf", "*.png", "*.jpg"] # 忽略非数据文件，提速
        )
        print(f"✅ {dataset_id} 下载完成！")
    except Exception as e:
        print(f"❌ {dataset_id} 遇到问题: {e}")

print(f"\n✨ 核心数据集下载尝试结束，存放于: {base_save_dir}")