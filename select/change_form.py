import json

# ==================== 在这里自定义路径 ====================
INPUT_FILE = "/home/zhouyan/share/cyh/CoT/select/refined_results/high_quality_math.json"
OUTPUT_FILE = "/home/zhouyan/share/cyh/CoT/select/refined_results/sft_train.jsonl"
# =========================================================

def main():
    # 读取原始 JSON
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 写入 JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in data:
            # 只提取 Swift 需要的字段
            clean_item = {
                "instruction": item["instruction"],
                "output": item["output"]
            }
            f.write(json.dumps(clean_item, ensure_ascii=False) + '\n')

    print(f"✅ 转换完成！")
    print(f"📊 共处理样本: {len(data)} 条")
    print(f"📁 已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
