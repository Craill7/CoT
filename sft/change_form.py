import pandas as pd
import json
import os
from tqdm import tqdm

# ================= 配置区 =================
INPUT_PATH = "/home/zhouyan/share/cyh/CoT/sample_data/train_sampled_3k.jsonl" 
OUTPUT_PATH = "/home/zhouyan/share/cyh/CoT/sft/swift_baseline_6k.jsonl"
# ==========================================

def main():
    print(f"📦 正在加载原始数据: {INPUT_PATH}")
    
    # 根据文件后缀自动选择读取方式
    if INPUT_PATH.endswith('.parquet'):
        df = pd.read_parquet(INPUT_PATH)
    elif INPUT_PATH.endswith('.jsonl'):
        df = pd.read_json(INPUT_PATH, lines=True)
    else:
        raise ValueError("只支持 .parquet 或 .jsonl 格式的输入文件！")
        
    print(f"📊 成功读取 {len(df)} 条原始题目数据，开始提取前双 CoT...")

    valid_count = 0
    skip_count = 0
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Expanding"):
            try:
                # 1. 提取 messages 字段 (用于获取 user 提问)
                messages = row.get('messages')
                if messages is None:
                    skip_count += 1
                    continue
                
                # 防御性编程：处理 pandas 读取时可能变成 numpy array 或 string 的情况
                if isinstance(messages, str):
                    messages = json.loads(messages)
                elif type(messages).__name__ == 'ndarray':
                    messages = messages.tolist()
                
                # 检查合法性：至少要有 user 消息
                if not isinstance(messages, list) or len(messages) < 1:
                    skip_count += 1
                    continue
                    
                # 提取用户的提问内容
                user_content = messages[0].get('content', '').strip()
                if not user_content:
                    skip_count += 1
                    continue

                # 2. 提取 generations 字段
                generations = row.get('generations')
                if generations is None:
                    skip_count += 1
                    continue
                    
                if isinstance(generations, str):
                    generations = json.loads(generations)
                elif type(generations).__name__ == 'ndarray':
                    generations = generations.tolist()
                
                # 3. 【核心对齐逻辑】只取前 2 条 CoT，与打分脚本保持绝对一致
                if isinstance(generations, list) and len(generations) > 0:
                    # 加上 [:2]，无论后面有几条，统统丢弃
                    for cot in generations[:2]: 
                        # 过滤掉空字符串或纯空格的 CoT
                        if not cot or not isinstance(cot, str) or not cot.strip():
                            continue
                            
                        clean_item = {
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": cot.strip()}
                            ]
                        }
                        
                        # 写入文件
                        f_out.write(json.dumps(clean_item, ensure_ascii=False) + '\n')
                        valid_count += 1
                else:
                    skip_count += 1

            except Exception as e:
                # 遇到任何解析极度畸形的数据直接跳过，保证流程不中断
                skip_count += 1
                continue

    print(f"\n🎉 双 CoT 提取与格式转换完成！")
    print(f"✅ 成功生成 SFT 训练数据: {valid_count} 条")
    print(f"⚠️ 跳过无效/畸形数据: {skip_count} 条")
    print(f"📁 已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
