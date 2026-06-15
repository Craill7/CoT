import os
import json
import torch
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

# --- 环境设置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_ifd_and_embedding(tokenizer, model, instruction, output, max_length):
    """计算指令跟随难度(IFD)分数和语义嵌入"""
    full_text = f"{instruction}\n{output}"
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
        loss = outputs.loss
        embeddings = outputs.hidden_states[-1]
        sentence_embedding = embeddings.mean(dim=1).float().cpu()
        
    return torch.exp(loss).item(), sentence_embedding

def process_quality_score(item):
    """多维加权积分制"""
    meta = item.get("metadata", {})
    is_correct = meta.get("is_correct", False)
    tags = meta.get("quality_tags", [])
    difficulty = meta.get("difficulty", "Unknown")
    
    if not is_correct:
        return 0.0
        
    base_score = 10.0
    tag_set = set(tags)
    valid_tags = {"Deep", "Present", "Exploratory", "Cohesive"}
    tag_score = sum(2.5 for t in tag_set if t in valid_tags)
    
    diff_multiplier = 1.0
    if difficulty == "Hard":
        diff_multiplier = 1.2
    elif difficulty == "Easy":
        diff_multiplier = 0.8
        
    return (base_score + tag_score) * diff_multiplier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./refined_results")
    parser.add_argument("--num_clusters", type=int, default=40)
    parser.add_argument("--k_ratio", type=float, default=0.8)
    # 【新增】强制重新计算的开关，无视旧缓存
    parser.add_argument("--force_recompute", action="store_true", help="强制忽略缓存重新计算")
    args = parser.parse_args()

    cache_file = os.path.join(args.output_dir, "embeddings_cache.pt")
    
    data_list = []
    embeddings = []
    
    # 1. 加载或重新计算
    if os.path.exists(cache_file) and not args.force_recompute:
        print(f"🏃 检测到缓存文件 {cache_file}...")
        checkpoint = torch.load(cache_file)
        data_list = checkpoint['data_list']
        embeddings = checkpoint['embeddings']
        
        # 【安全校验】检查缓存里的数据是否有 GT 字段
        if len(data_list) > 0 and 'solution' not in data_list[0]:
            print("⚠️ 警告：检测到旧版缓存缺失 solution/answer 字段！")
            print("⚠️ 请终止程序，加上 --force_recompute 参数重新运行，或者手动删除 embeddings_cache.pt！")
            return
            
        print(f"✅ 成功加载 {len(data_list)} 条缓存数据 (字段校验通过)！")
    else:
        print("🚀 加载模型中 (将重新计算 IFD 与 Embeddings)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        os.makedirs(args.output_dir, exist_ok=True)
        
        print("📊 开始计算 PPL 和 Embedding...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                item = json.loads(line)
                
                # 【显式提取】确保无论输入格式如何，我们都显式抓取这两个字段
                # 如果前序脚本没有这俩字段，给个空字符串兜底防报错
                item['solution'] = item.get('solution', '')
                item['answer'] = item.get('answer', '')
                
                ppl, emb = get_ifd_and_embedding(tokenizer, model, item['instruction'], item['output'], 2048)
                
                item['IFD_Score'] = ppl
                item['weighted_score'] = process_quality_score(item)
                data_list.append(item)
                embeddings.append(emb)
                
        print("💾 正在保存特征缓存 (包含完整的 solution 和 answer)...")
        torch.save({'data_list': data_list, 'embeddings': embeddings}, cache_file)
        
        del model
        del tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # 2. 聚类
    print(f"🧩 正在进行 KMeans 聚类 (簇数量: {args.num_clusters})...")
    features = torch.cat(embeddings, 0).numpy()
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, n_init='auto').fit(features)
    for idx, label in enumerate(kmeans.labels_):
        data_list[idx]['Class'] = int(label)

    # 3. 按簇比例筛选
    print("⚖️ 执行多维加权比例筛选...")
    final_high = []
    final_low = []
    
    clusters = {}
    for item in data_list:
        c_id = item['Class']
        if c_id not in clusters: clusters[c_id] = []
        clusters[c_id].append(item)

    for c_id, items in clusters.items():
        items.sort(key=lambda x: (x['weighted_score'], x['IFD_Score']), reverse=True)
        target_count = int(len(items) * args.k_ratio)
        final_high.extend(items[:target_count])
        final_low.extend(items[target_count:])

    # 4. 保存最终结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    out_path_high = os.path.join(args.output_dir, "high_quality_math.json")
    with open(out_path_high, 'w', encoding='utf-8') as f:
        json.dump(final_high, f, ensure_ascii=False, indent=4)
        
    out_path_low = os.path.join(args.output_dir, "low_quality_math.json")
    with open(out_path_low, 'w', encoding='utf-8') as f:
        # 写入 Low Quality 时，它们身上已经牢牢绑定了 solution 和 answer
        json.dump(final_low, f, ensure_ascii=False, indent=4)
    
    scores_high = [item['weighted_score'] for item in final_high]
    scores_low = [item['weighted_score'] for item in final_low] if final_low else [0]
    
    print(f"\n✅ 筛选完成！")
    print(f"🌟 保留高质量样本: {len(final_high)} 条 (平均得分: {np.mean(scores_high):.2f})")
    print(f"🗑️ 剔除低质量样本: {len(final_low)} 条 (平均得分: {np.mean(scores_low):.2f})")
    print(f"📁 高质量数据已保存至: {out_path_high}")
    print(f"📁 低质量数据已保存至: {out_path_low}")

if __name__ == "__main__":
    main()