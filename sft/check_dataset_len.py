import json
from transformers import AutoTokenizer
from tqdm import tqdm

# 配置路径
model_path = "/nfsdata-117/model/Qwen/Qwen2.5-3B"
dataset_path = "/home/zhouyan/share/cyh/CoT/sample_data/train_sampled_3k.jsonl"

def get_all_text(obj):
    """递归提取 JSON 对象中所有的文本内容"""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        return "".join([get_all_text(item) for item in obj])
    elif isinstance(obj, dict):
        return "".join([get_all_text(value) for value in obj.values()])
    return ""

def check_len():
    print(f"正在加载分词器: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    lengths = []
    print(f"正在分析数据集: {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                # 无论你的 key 叫 content, value 还是 output，统统抓取
                full_text = get_all_text(data)
                
                if not full_text:
                    continue
                    
                tokens = tokenizer.encode(full_text)
                lengths.append(len(tokens))
            except Exception as e:
                continue

    if not lengths:
        print("❌ 错误：依然没搜集到任何文本，请检查 JSONL 格式！")
        return

    lengths.sort()
    max_l = lengths[-1]
    avg_l = sum(lengths) / len(lengths)
    p95 = lengths[int(len(lengths) * 0.95)]
    p99 = lengths[int(len(lengths) * 0.99)]

    print("\n" + "="*40)
    print(f"📊 数据集 Token 长度统计结果 (修正版):")
    print("-" * 40)
    print(f"🔹 样本总数: {len(lengths)}")
    print(f"🔹 最大长度: {max_l}")
    print(f"🔹 平均长度: {avg_l:.2f}")
    print(f"🔹 95% 分位数: {p95}")
    print(f"🔹 99% 分位数: {p99}")
    print("="*40)
    
    # 动态建议
    suggested = ((p95 // 512) + 1) * 512
    print(f"\n💡 建议: --max_length 设为 {int(suggested)}")

if __name__ == "__main__":
    check_len()