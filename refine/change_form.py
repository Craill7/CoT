import json
import os

# ================= 配置区 =================
# 你提纯后的高质量数据集路径
INPUT_FILE = "/home/zhouyan/share/cyh/CoT/refine/refined_results/all_high_quality_math.jsonl" 
# SWIFT SFT 微调目标文件路径
OUTPUT_FILE = "/home/zhouyan/share/cyh/CoT/refine/refined_results/all_high_quality_math_swift.jsonl"

# 推荐添加 System Prompt，用于规范模型的输出行为
# 如果你想测试模型在无 system prompt 下的泛化能力，可以将此设为空字符串 ""
SYSTEM_PROMPT = "You are an expert mathematical reasoning assistant. Please solve the problem step-by-step, thinking logically and carefully. Enclose your final answer within \\boxed{}."
# ==========================================

def convert_to_swift_format():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        return

    raw_data = []
    print(f"📦 正在加载原始数据: {INPUT_FILE}")
    
    # 兼容 JSON 和 JSONL 的读取
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            raw_data = json.loads(content)
        else:
            raw_data = [json.loads(line) for line in content.split('\n') if line.strip()]

    print(f"🔍 成功加载 {len(raw_data)} 条原始数据，正在转换格式...")

    converted_data = []
    for item in raw_data:
        # 提取核心指令和思维链输出
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()

        # 过滤掉空数据
        if not instruction or not output:
            continue

        # 组装为 SWIFT 支持的 query-response 格式
        swift_item = {
            "query": instruction,
            "response": output
        }
        
        # 只有在设置了 SYSTEM_PROMPT 时才添加该字段
        if SYSTEM_PROMPT:
            swift_item["system"] = SYSTEM_PROMPT

        converted_data.append(swift_item)

    # 写入 JSONL 格式（SWIFT 对 JSONL 读取效率最高）
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in converted_data:
            # ensure_ascii=False 保证中文等非 ASCII 字符正常显示
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "="*50)
    print("🏆 转换完成！")
    print(f"✅ 有效转换条数: {len(converted_data)}")
    print(f"📁 SWIFT 专用数据集已保存至: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    convert_to_swift_format()