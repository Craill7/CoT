import json
import requests
import re
import os
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 全局配置区 =================
INPUT_PATH = "/home/zhouyan/share/cyh/CoT/3k_test_data/data_sample_3k.jsonl"
OUTPUT_PATH = "/home/zhouyan/share/cyh/CoT/data_shorten"

API_URL = "http://localhost:8001/v1/chat/completions"
API_MODEL_NAME = "Qwen2.5-3B-Instruct" 

MAX_WORKERS = 4
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

def process_item(item):
    # 局部统计字典，防止多线程冲突
    local_stats = {
        "checked": 0,
        "compressed": 0,
        "pre_len": [],
        "post_len": []
    }
    
    try:
        problem = item.get('problem', '')
        generations = item.get('generations', [])
        completes = item.get('is_reasoning_complete', [])
        corrects = item.get('correctness_math_verify', [])
        
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
                # 记录压缩前的长度
                original_len = len(cot)
                folded_cot = fold_long_cot(cot)
                
                try:
                    compressed_cot = call_llm_to_compress(problem, folded_cot, reason)
                    new_generations.append(compressed_cot)
                    new_completes.append(is_comp)
                    new_corrects.append(is_corr)
                    
                    # 记录成功压缩后的统计数据
                    local_stats["compressed"] += 1
                    local_stats["pre_len"].append(original_len)
                    local_stats["post_len"].append(len(compressed_cot))
                    
                except Exception as e:
                    pass
        
        if len(new_generations) == 0:
            return None, local_stats
            
        item['generations'] = new_generations
        item['is_reasoning_complete'] = new_completes
        item['correctness_math_verify'] = new_corrects
        
        return item, local_stats
        
    except Exception as e:
        return None, local_stats

def main():
    print(f"📦 正在读取原始数据: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
        
    print(f"⚖️ 开始执行残次品剔除与 AI 压缩手术 (总题目数: {len(all_data)})...")
    
    # 全局统计指标
    global_stats = {
        "total_cots_checked": 0,
        "total_cots_compressed": 0,
        "all_pre_lengths": [],
        "all_post_lengths": []
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_item, item): item for item in all_data}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 Optimizing"):
                result_item, result_stats = future.result()
                
                # 累加局部统计到全局
                global_stats["total_cots_checked"] += result_stats["checked"]
                global_stats["total_cots_compressed"] += result_stats["compressed"]
                global_stats["all_pre_lengths"].extend(result_stats["pre_len"])
                global_stats["all_post_lengths"].extend(result_stats["post_len"])
                
                if result_item is not None:
                    f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    f_out.flush()

    # --- 打印最终手术报告 ---
    print("\n" + "="*55)
    print("🏥 AI 逻辑微创手术结果报告 (Before & After)")
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
        print(f"🔥 综合压缩率 (空间释放): {compress_ratio:.2f}% (脂肪被狠狠抽走了！)")
    else:
        print("-" * 55)
        print("🤷‍♂️ 本次运行未发现需要压缩的冗余 CoT。")
    print("="*55)
    print(f"📁 优化后的数据集已存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()