import json

# 这是你流水线最早的输入文件（3k抽样底包）
RAW_INPUT_FILE = "/home/zhouyan/share/cyh/CoT/3k_test_data/data_sample_3k.jsonl"

# 我们用题目的前几个独特单词作为检索指纹
TARGET_PROBLEMS = {
    "Case_1": "Determine all possibilities to specify a natural number",
    "Case_2": "Given the sets $A=\\left\\{x \\mid \\log _{2}(x-1)<1\\right\\}",
    "Case_5": "Let $a, b \\in \\mathbf{N}^{*},(a, b)=1$"
}

def main():
    found_cases = {}
    
    print(f"📦 正在扫描原始底层数据: {RAW_INPUT_FILE}")
    
    try:
        with open(RAW_INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                item = json.loads(line)
                problem_text = item.get('problem', '')
                
                # 遍历看是否命中了我们的通缉犯
                for case_name, keyword in TARGET_PROBLEMS.items():
                    if keyword in problem_text:
                        found_cases[case_name] = {
                            "problem": problem_text[:100] + "...",
                            "raw_answer": item.get('answer', '未找到 answer 字段')
                        }
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 打印法庭证据
    print("\n" + "="*60)
    print("🕵️ 原始数据集 GT 溯源验尸报告")
    print("="*60)
    
    for case_name in sorted(TARGET_PROBLEMS.keys()):
        data = found_cases.get(case_name)
        if data:
            print(f"🩸 {case_name}")
            print(f"❓ 原题开头: {data['problem']}")
            print(f"🎯 最原始的 Answer 字段值: {data['raw_answer']}")
            print("-" * 60)
        else:
            print(f"⚠️ {case_name} 未在当前文件中找到！")

if __name__ == "__main__":
    main()