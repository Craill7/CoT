import os
import json
import glob
import requests
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 全局配置区 =================
INPUT_DIR = "/mdr0/user/quantaalpha/zhouyan/cyh/datasets/cleaned_data/OpenR1-Math-220k"
OUTPUT_DIR = "/mdr0/user/quantaalpha/zhouyan/cyh/datasets/shorten_data/OpenR1-Math-220k/optimized_jsonl"

API_URL = "http://localhost:8001/v1/chat/completions"
API_MODEL_NAME = "Qwen2.5-3B-Instruct" 

MAX_WORKERS = 16       # A800 跑 3B 模型，并发可以开大点，榨干吞吐
REQUEST_TIMEOUT = 120  
# ==========================================

def get_logic_skeleton(text):
    text = text.lower()
    text = re.sub(r'[0-9a-zA-Z\+\-\*\/\=\(\)\[\]\{\}\^\$\\\.\,\:\;\_]', ' ', text)
    return ' '.join(text.split())

def analyze_r1_cot_features(cot_text):
    if len(cot_text) < 2000:
        return True, "Valid"
        
    lines = [line.strip() for line in cot_text.split('\n') if len(line.strip()) > 30]
    if len(lines) > 20:
        blocks = ['|'.join(lines[i:i+3]) for i in range(len(lines)-2)]
        block_counts = Counter(blocks)
        most_common_block = block_counts.most_common(1)
        if most_common_block and most_common_block[0][1] >= 5:
            return False, "Physical Loop"

    skeleton = get_logic_skeleton(cot_text)
    words = skeleton.split()
    if len(words) > 50:
        ngrams = [' '.join(words[i:i+10]) for i in range(len(words)-9)]
        ngram_counts = Counter(ngrams)
        most_common = ngram_counts.most_common(1)
        if most_common and most_common[0][1] >= 6:
            return False, "Skeleton Loop"

    return True, "Valid"

def fold_long_cot(cot_text, max_length=4000):
    if len(cot_text) <= max_length:
        return cot_text
        
    lines = cot_text.split('\n')
    head_lines = lines[:40]
    tail_lines = lines[-40:]
    omitted = len(lines) - 80
    
    return "\n".join(head_lines) + f"\n\n...[系统提示：已物理折叠中间 {omitted} 行深陷循环的冗余文本]...\n\n" + "\n".join(tail_lines)

def call_llm_to_compress(problem, folded_cot, loop_type):
    prompt = f"""你是一个专门处理大模型微调数据的“文本整形专家”。
下面的数学思维链（CoT）在中间部分陷入了死循环（诊断标签：{loop_type}），且为了防止过长已经被物理折叠。
你的任务是：梳理这段文本，将折叠的死循环部分化简为一句平滑的过渡逻辑，将前后文完美缝合。

【绝对规则（违反则任务失败）】：
1. 仅仅压缩和修复死循环部分，绝对不要改变原始的解题思路和最终答案（即使它是错的，也要照着错的写）。
2. 绝对禁止从头做题，不要输出多余的解释、寒暄或分析。
3. 必须保留并闭合原有的 <think> 和 </think> 标签。

题目:
{problem}

原始带折叠的冗余 CoT:
{folded_cot}

请直接输出缝合压缩后的完整 CoT："""

    payload = {
        "model": API_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1 
    }
    
    r = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content'].strip()

def safe_tolist(val):
    """安全地将 Parquet 读取的 Numpy Array 转换为 Python List"""
    if val is None:
        return []
    if hasattr(val, 'tolist'):
        return val.tolist()
    return list(val)

def process_item(item):
    local_stats = {"checked": 0, "compressed": 0, "pre_len": [], "post_len": []}
    
    try:
        problem = item.get('problem', '')
        
        # 安全解包 Parquet 数组
        generations = safe_tolist(item.get('generations', []))
        completes = safe_tolist(item.get('is_reasoning_complete', []))
        corrects = safe_tolist(item.get('correctness_math_verify', []))
        
        new_generations = []
        new_completes = []
        new_corrects = []
        
        for i in range(len(generations)):
            cot = generations[i]
            is_comp = completes[i] if i < len(completes) else False
            is_corr = corrects[i] if i < len(corrects) else False
            
            if not is_comp:
                continue
                
            local_stats["checked"] += 1
            is_valid, reason = analyze_r1_cot_features(cot)
            
            if reason == "Valid":
                new_generations.append(cot)
                new_completes.append(is_comp)
                new_corrects.append(is_corr)
            else:
                original_len = len(cot)
                folded_cot = fold_long_cot(cot)
                
                try:
                    compressed_cot = call_llm_to_compress(problem, folded_cot, reason)
                    new_generations.append(compressed_cot)
                    new_completes.append(is_comp)
                    new_corrects.append(is_corr)
                    
                    local_stats["compressed"] += 1
                    local_stats["pre_len"].append(original_len)
                    local_stats["post_len"].append(len(compressed_cot))
                except Exception:
                    pass
        
        if len(new_generations) == 0:
            return None, local_stats
            
        item['generations'] = new_generations
        item['is_reasoning_complete'] = new_completes
        item['correctness_math_verify'] = new_corrects
        
        return item, local_stats
        
    except Exception:
        return None, local_stats

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    file_pattern = os.path.join(INPUT_DIR, "train-0000*-of-00010.parquet")
    parquet_files = sorted(glob.glob(file_pattern))
    
    if not parquet_files:
        print(f"❌ 未找到任何 Parquet 文件，请检查路径: {INPUT_DIR}")
        return

    print(f"📂 发现 {len(parquet_files)} 个分片文件，准备全量手术...")

    global_stats = {
        "total_cots_checked": 0,
        "total_cots_compressed": 0,
        "all_pre_lengths": [],
        "all_post_lengths": []
    }

    # 按文件分片遍历
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        # 构造输出文件名: train-00000-of-00010.parquet -> optimized_train-00000.jsonl
        out_name = "optimized_" + file_name.split('-of-')[0] + ".jsonl"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # 【断点续跑】如果该分片已存在且有内容，则跳过
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"⏭️  检测到已完成分片 {out_name}，跳过...")
            continue
            
        print(f"\n🔄 正在加载数据分片: {file_name}")
        df = pd.read_parquet(file_path)
        # 将 DataFrame 转为字典列表
        data_chunk = df.to_dict('records')
        
        print(f"⚖️ 开始处理 {file_name} (包含 {len(data_chunk)} 题)...")
        
        with open(out_path, 'w', encoding='utf-8') as f_out:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_item, item): item for item in data_chunk}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"🚀 处理 {file_name}"):
                    result_item, result_stats = future.result()
                    
                    global_stats["total_cots_checked"] += result_stats["checked"]
                    global_stats["total_cots_compressed"] += result_stats["compressed"]
                    global_stats["all_pre_lengths"].extend(result_stats["pre_len"])
                    global_stats["all_post_lengths"].extend(result_stats["post_len"])
                    
                    if result_item is not None:
                        f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
        print(f"✅ 分片 {file_name} 处理完毕，结果存入 {out_name}")
        
        # 释放内存
        del df
        del data_chunk
        import gc
        gc.collect()

    # --- 打印最终宏观手术报告 ---
    print("\n" + "="*55)
    print("🏥 AI 逻辑微创手术 全量报告 (Before & After)")
    print("-" * 55)
    print(f"🔸 实际扫描有效 CoT 总数: {global_stats['total_cots_checked']:,} 条")
    print(f"🔸 成功实施压缩手术的 CoT: {global_stats['total_cots_compressed']:,} 条")
    
    if global_stats['total_cots_compressed'] > 0:
        pre_lens = global_stats['all_pre_lengths']
        post_lens = global_stats['all_post_lengths']
        
        pre_avg = sum(pre_lens) // len(pre_lens)
        pre_max = max(pre_lens)
        post_avg = sum(post_lens) // len(post_lens)
        post_max = max(post_lens)
        compress_ratio = (1 - (sum(post_lens) / sum(pre_lens))) * 100
        
        print("-" * 55)
        print("📉 瘦身数据对比 (针对被压缩的数据)：")
        print(f"   [手术前] 平均字符数: {pre_avg:,} | 最大字符数: {pre_max:,}")
        print(f"   [手术后] 平均字符数: {post_avg:,} | 最大字符数: {post_max:,}")
        print("-" * 55)
        print(f"🔥 综合压缩率 (空间释放): {compress_ratio:.2f}%")
    print("="*55)
    print(f"📁 优化后的 10 个 JSONL 分片已全生存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()