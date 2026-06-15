import json
import os

# ================= 配置区 =================
# 替换为你实际的低质量数据集路径
INPUT_PATH = "/home/zhouyan/share/cyh/CoT/select/refined_results/low_quality_math.json" 
# ==========================================

def main():
    print(f"📦 正在加载低质量样本数据: {INPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 找不到文件，请检查路径: {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    if total_samples == 0:
        print("⚠️ 数据集为空！")
        return

    incorrect_count = 0
    correct_count = 0

    for item in data:
        # 获取正确性标签 (适配我们之前统一定义的 metadata 结构)
        meta = item.get("metadata", {})
        is_correct = meta.get("is_correct", False)
        
        if not is_correct:
            incorrect_count += 1
        else:
            correct_count += 1

    # --- 打印分析报告 ---
    print("\n" + "="*50)
    print("🔬 低质量样本 (Low-Quality Pool) 错误率体检报告")
    print("-" * 50)
    print(f"🔸 扫描样本总数: {total_samples:,} 条")
    print("-" * 50)
    
    incorrect_ratio = (incorrect_count / total_samples) * 100
    correct_ratio = (correct_count / total_samples) * 100
    
    print(f"❌ 最终算错的 CoT (优质 DPO 负样本): {incorrect_count:,} 条 ({incorrect_ratio:.2f}%)")
    print(f"✅ 算对但低质的 CoT (待压缩 Refine 样本): {correct_count:,} 条 ({correct_ratio:.2f}%)")
    print("="*50)
    
    if correct_count > 0:
        print(f"\n💡 结论：接下来你需要对这 {correct_count} 条“算对但低质”的数据进行大模型微创压缩手术。")

if __name__ == "__main__":
    main()