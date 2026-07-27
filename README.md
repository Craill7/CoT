# CoT — Long Chain-of-Thought Refinement Pipeline

将冗长数学推理链压缩为精炼推理，保留正确性同时提升可读性。

## 模块

| 模块 | 用途 |
|---|---|
| `pipeline/` | 主编排器：压缩 → 筛选 → 验证 → 重写 |
| `select/` | CoT 聚类 + Pairwise LLM 评判筛选 |
| `refine/` | CoT 精炼压缩核心（依赖标注 → 原子分段 → 块标注 → 全局验证） |

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 vLLM 推理服务
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model --port 8000

# 3. 运行完整 pipeline
python -m pipeline.run \
    --input data.jsonl \
    --output results/ \
    --llm-url http://localhost:8000/v1 \
    --model default
```

## 依赖

- Python 3.12+
- vLLM 0.19+（推理）
- math_verify + latex2sympy2_extended（数学判分）
- PyTorch 2.10+（CUDA 12.0）
