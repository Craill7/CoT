# Benchmark 测评标化指南

> 最后更新：2026-06-25  
> 基于 compress-v3-lora 实验全流程整理

---

## 1. 环境与路径

### 环境

| 用途 | 环境 | 激活命令 |
|------|------|----------|
| 测评 | `eval` | `source /mdr5/guest/users/zhouyan/share/cyh/envs/eval/bin/activate` |
| 训练 | `swift` | `source /mdr5/guest/users/zhouyan/share/cyh/envs/swift/bin/activate` |
| vLLM 推理 | `cot_opt` | `source /mdr5/guest/users/zhouyan/share/cyh/envs/cot_opt/bin/activate` |

### 关键路径

| 用途 | 路径 |
|------|------|
| OpenCompass 框架 | `/ky200t/guest/users/zhouyan/cyh/opencompass` |
| 测评数据 | `/ky200t/guest/users/zhouyan/cyh/benchmarks/` |
| Merge 模型 | `/ky200t/guest/users/zhouyan/cyh/mergemodels/<name>/` |
| 实验目录 | `CoT/experiments/<exp-name>/` |
| 基座模型 | `/ky200t/models/Qwen2.5-3B`、`/ky200t/models/Qwen2.5-32B-Instruct` |

---

## 2. 实验目录结构

每个实验在 `CoT/experiments/<name>/` 下独立存放，标准结构：

```
experiments/<name>/
├── bench_config.py       # OpenCompass 数据集 + 模型配置
├── bench_cmd.sh          # 测评启动命令（含 CUDA/环境变量）
├── train_cmd.sh          # 训练命令记录（SFT / LoRA）
└── results/              # 测评结果（跑完后从 opencompass/outputs/ 复制）
```

---

## 3. 标准测评数据集

### 3.1 数据集配置模板 (`bench_config.py`)

```python
datasets = [
    # ── MATH-500 ──
    dict(
        abbr='MATH-500',
        type='opencompass.datasets.CustomDataset',
        path='/ky200t/guest/users/zhouyan/cyh/benchmarks/HuggingFaceH4_MATH-500/test.jsonl',
        reader_cfg=dict(input_columns=['problem'], output_column='answer'),
        eval_cfg=dict(evaluator=dict(type='opencompass.evaluator.MATHVerifyEvaluator')),
        infer_cfg=dict(
            inferencer=dict(type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                type='opencompass.openicl.icl_prompt_template.PromptTemplate',
                template=dict(round=[
                    dict(role='HUMAN',
                         prompt='Problem:\n{problem}\nPlease reason step by step and put your final answer within \\boxed{}.'),
                ]),
            ),
            retriever=dict(type='opencompass.openicl.icl_retriever.ZeroRetriever'),
        ),
    ),

    # ── AMC23 ──
    dict(
        abbr='AMC23',
        type='opencompass.datasets.CustomDataset',
        path='/ky200t/guest/users/zhouyan/cyh/benchmarks/math-ai_amc23/test-00000-of-00001.jsonl',
        reader_cfg=dict(input_columns=['question'], output_column='answer'),
        eval_cfg=dict(evaluator=dict(type='opencompass.evaluator.MATHVerifyEvaluator')),
        infer_cfg=dict(
            inferencer=dict(type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                type='opencompass.openicl.icl_prompt_template.PromptTemplate',
                template=dict(round=[
                    dict(role='HUMAN',
                         prompt='Problem:\n{question}\nPlease reason step by step and put your final answer within \\boxed{}.'),
                ]),
            ),
            retriever=dict(type='opencompass.openicl.icl_retriever.ZeroRetriever'),
        ),
    ),

    # ── AIME2025 ──
    dict(
        abbr='AIME2025',
        type='opencompass.datasets.CustomDataset',
        path='/ky200t/guest/users/zhouyan/cyh/benchmarks/opencompass_AIME2025/aime2025.jsonl',
        reader_cfg=dict(input_columns=['question'], output_column='answer'),
        eval_cfg=dict(evaluator=dict(type='opencompass.evaluator.MATHVerifyEvaluator')),
        infer_cfg=dict(
            inferencer=dict(type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                type='opencompass.openicl.icl_prompt_template.PromptTemplate',
                template=dict(round=[
                    dict(role='HUMAN',
                         prompt='Problem:\n{question}\nPlease reason step by step and put your final answer within \\boxed{}.'),
                ]),
            ),
            retriever=dict(type='opencompass.openicl.icl_retriever.ZeroRetriever'),
        ),
    ),
]
```

### 3.2 数据集字段约定

| 数据集 | 输入字段 | 答案字段 | 题型 |
|--------|---------|---------|------|
| MATH-500 | `problem` | `answer` | 竞赛数学（多领域） |
| AMC23 | `question` | `answer` | AMC 2023 竞赛题 |
| AIME2025 | `question` | `answer` | AIME 2025 邀请赛 |

> **注意**：`input_columns` 和 prompt 中的 `{problem}`/`{question}` 占位符必须与数据文件的字段名一致。

### 3.3 评测器

全部使用 `MATHVerifyEvaluator`，它从模型输出中提取 `\boxed{...}` 内容与标准答案比对。

---

## 4. 模型配置

### 4.1 模型配置模板（vLLM 推理）

```python
models = [
    dict(
        abbr='<模型简称>',                      # 用于输出目录命名
        type='opencompass.models.VLLM',
        path='/ky200t/guest/users/zhouyan/cyh/mergemodels/<模型文件夹名>',
        batch_size=16,
        max_out_len=4096,
        generation_kwargs=dict(temperature=0),
        model_kwargs=dict(
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=1,
        ),
        run_cfg=dict(num_gpus=1),
    ),
]
```

### 4.2 关键参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `batch_size` | vLLM 批处理大小 | 16 |
| `max_out_len` | 最大生成 token 数 | 4096（数学推理足够） |
| `temperature` | 采样温度 | 0（贪心解码） |
| `gpu_memory_utilization` | 显存占用比例 | 0.5（给 KV cache 留空间） |
| `enforce_eager` | 禁用 CUDA graph | True（避免显存不足） |
| `tensor_parallel_size` | 张量并行数 | 1（小模型单卡足够） |

### 4.3 模型路径约定

| 阶段 | 路径 |
|------|------|
| 训练产出 | `/ky200t/guest/users/zhouyan/cyh/checkpoints/<name>/v0-YYYYMMDD-HHMMSS/checkpoint-N` |
| Merge 后 | `/ky200t/guest/users/zhouyan/cyh/mergemodels/<name>/` |
| 基座模型 | `/ky200t/models/Qwen2.5-3B` |

> **纪律**：merge 模型用完即删（`mergemodels/` 是大文件临时存放点）。

---

## 5. 启动测评

### 5.1 启动命令 (`bench_cmd.sh`)

```bash
#!/bin/bash
source /mdr5/guest/users/zhouyan/share/cyh/envs/eval/bin/activate
cd /ky200t/guest/users/zhouyan/cyh/opencompass

VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_V1=0 \
CUDA_VISIBLE_DEVICES=<GPU_ID> \
python run.py \
    --models <模型配置名> \
    --datasets <数据集配置名> \
    --work-dir outputs/<实验名>
```

### 5.2 环境变量说明

| 变量 | 值 | 原因 |
|------|-----|------|
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | 避免 vLLM 多进程 fork 冲突 |
| `VLLM_USE_V1` | `0` | 使用 vLLM V0 API（兼容 opencompass） |
| `CUDA_VISIBLE_DEVICES` | GPU 编号 | 指定运行 GPU |

### 5.3 配置发现机制

OpenCompass 通过 `--models` 和 `--datasets` 参数指定**配置文件名**（不含 `.py`）：

- 模型配置放在 `<opencompass>/configs/models/` 下
- 数据集配置放在 `<opencompass>/configs/datasets/` 下
- 也可以用 `bench_config.py` 放在实验目录，通过 `--config` 直接引用

### 5.4 实际操作

实际做法是写一个 `bench_config.py`（含 datasets + models + work_dir），放在实验目录，然后 copy 到 opencompass 的 configs 目录注册：

```bash
# 方式一：注册为独立配置文件
cp bench_config.py /ky200t/guest/users/zhouyan/cyh/opencompass/configs/
python run.py --models my_config --datasets my_config

# 方式二：直接用 run.py 的 --config 参数
python run.py --config /path/to/bench_config.py
```

推荐方式二，配置文件随实验目录一起版本管理。

---

## 6. 运行与监控

### 6.1 tmux 启动

```bash
tmux new-window -t cyh -n bench 'cd /mdr5/guest/users/zhouyan/share/cyh/CoT/experiments/<exp> && bash bench_cmd.sh 2>&1 | tee bench.log'
```

### 6.2 监控

```bash
# GPU 状态
nvidia-smi

# 输出目录进度
ls outputs/<实验名>/

# 实时结果
tail -f outputs/<实验名>/<timestamp>/summary/*.csv
```

### 6.3 预计耗时

| 模型 | 数据集 | 预计 |
|------|--------|------|
| Qwen2.5-3B | MATH-500 (500题) | ~15 min |
| Qwen2.5-3B | AMC23 (150题) | ~5 min |
| Qwen2.5-3B | AIME2025 (30题) | ~2 min |
| **3 数据集合计** | | **~22 min** |

---

## 7. 结果处理

### 7.1 目录结构

测评完成后，opencompass 产出：

```
opencompass/outputs/<实验名>/<timestamp>/
├── summary/
│   └── *.csv           # 各数据集得分汇总
├── predictions/
│   └── */               # 模型预测输出
└── logs/
```

### 7.2 结果归档

```bash
# 复制到实验目录
cp -r /ky200t/guest/users/zhouyan/cyh/opencompass/outputs/<实验名>/<timestamp>/ \
     CoT/experiments/<exp>/results/
```

### 7.3 关键指标

从 `summary/*.csv` 中提取：

| 指标 | 含义 |
|------|------|
| `accuracy` | 最终答案正确率（MATHVerify） |
| `num_samples` | 有效样本数 |

---

## 8. 完整实验流程 Checklist

### 8.1 训练前

- [ ] 准备训练数据（JSONL，`instruction` + `output` 格式）
- [ ] 确认 GPU 空闲（`nvidia-smi`）
- [ ] 激活 swift 环境
- [ ] 确定超参（lr, batch_size, LoRA rank, max_length 等）

### 8.2 训练

- [ ] 启动 tmux 窗口跑训练
- [ ] 确认 loss 收敛（`tail -f` 训练日志）
- [ ] 记录 checkpoint 路径和步数

### 8.3 Merge

```bash
# LoRA merge 命令
python -m swift.export \
    --adapters <checkpoint_path> \
    --merge_lora true \
    --output_dir /ky200t/guest/users/zhouyan/cyh/mergemodels/<name>
```

### 8.4 测评

- [ ] 准备 `bench_config.py`（数据集 + 模型配置）
- [ ] 准备 `bench_cmd.sh`（启动命令）
- [ ] 确认 merge 模型路径正确
- [ ] 确认 GPU 空闲
- [ ] 激活 eval 环境
- [ ] 启动 tmux 窗口跑测评
- [ ] 确认结果正常（无 OOM / 无崩溃）

### 8.5 收尾

- [ ] 结果复制到 `experiments/<exp>/results/`
- [ ] 删除 `mergemodels/<name>/`（释放空间）
- [ ] 记录关键指标到实验 README
- [ ] GPU 清理（`fuser -k /dev/nvidia<N>` 或确认进程退出）
- [ ] Git commit + push

---

## 9. 常见问题

### 9.1 vLLM 启动失败

**现象**: `Engine core initialization failed`

**排查**:
```bash
# 检查 GPU 是否有残留进程
fuser -v /dev/nvidia<N>

# 检查显存
nvidia-smi

# 降低 gpu_memory_utilization（如 0.5 → 0.4）
```

### 9.2 OOM / KV cache 不足

**现象**: `ValueError: KV cache is larger than available memory`

**解决**: 降 `max_model_len` 或升 `gpu_memory_utilization`

### 9.3 评测结果为 0

**排查**:
- 数据字段名是否正确（`problem` vs `question`）
- prompt 占位符 `{problem}` vs `{question}` 是否与字段匹配
- 模型路径是否正确
- 模型是否正确 merge

### 9.4 进程残留

```bash
# 查 GPU 占用
fuser -v /dev/nvidia*

# 杀进程
fuser -k /dev/nvidia<N>

# 验证
nvidia-smi | grep -A2 "Processes"
```

---

## 10. 参考实验

| 实验 | 目录 | 模型 | 数据集 | 结果 |
|------|------|------|--------|------|
| compress-v3-lora | `experiments/compress-v3-lora/` | Qwen2.5-3B + LoRA r=8 | MATH-500, AMC23, AIME2025 | 待查 |

---

## 附录：GPU 分配约定

| GPU | 允许使用 | 说明 |
|-----|---------|------|
| 0-3 | ✅ | 可用，但需确认无人占用 |
| 4-7 | ✅ | 可用 |
| 全卡 | — | 跑前 `nvidia-smi` 确认，空挂（0% util + 有显存）可清理 |

> **黄金法则**：`nvidia-smi` 看利用率，0% util + 高显存 = 空挂可杀；>50% util = 在跑别动。
