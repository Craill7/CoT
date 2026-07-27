# CoT Pipeline V2.1.5 架构与主流程

本文描述仓库中 `pipeline_v2/` 的当前实现。它是对既有 `select_v2` 与
Compress V4.2 的**版本化编排层**：复用现有选择、评分、分块和标注能力，
不覆盖既有 V4.2 产物。

## 1. 目标与边界

V2.1.5 面向 Long-CoT 数据的闭环筛选与修复。它把每条样本推进到明确的
`accepted` 或 `rejected` 终态，并保留全过程的 JSONL lineage。首版的上界是
每条样本最多一次受限改写；不会进行无限重试或自由重写整条 CoT。

核心链路如下：

```text
输入规范化
  -> 可插拔候选选择
  -> 单条全局评分与聚类排序
  -> high / low 路由

high: 语义答案门禁 -> accepted 或 answer_audit -> rejected

low: Compress V4.2 严格分块/标注/图构建
  -> block-to-block 依赖诊断
  -> closed: supporting-block 裁剪
  -> open + repairable: 结构化 patch 改写一次
  -> 重新分块、标注、构图、再诊断
  -> 确定性 Verifier -> LLM Verifier
  -> accepted / rejected
```

## 2. 主入口与装配

入口是 `python -m pipeline_v2.run`，代码在 `pipeline_v2/run.py`。

`build_pipeline()` 根据 `PipelineConfig` 装配实际组件：

| 职责 | 实现 | 说明 |
| --- | --- | --- |
| 候选路由 | `SelectionRouter` | `auto` 时单候选走 identity，多候选走 pairwise |
| 候选选择 | `IdentitySelector` / `PairwiseSelector` | pairwise 复用 `select_v2` judge、验证标签和 penalty |
| 全局评分 | `ReuseOrAbsoluteGlobalScorer` | 对单个选中 CoT 输出绝对质量分、难度与风险标签 |
| 聚类 | `ScoreRatioClusterer` / `IfdKMeansClusterer` | 前者用于轻量 smoke，后者复用 IFD、embedding、KMeans 与簇内 Top-K |
| 高质量答案判断 | `LLMAnswerJudge` | 按答案类型做语义比较，而非单纯字符串相等 |
| Compress 适配 | `V42Analyzer` | 复用 Compress V4.2，但注入 V2 严格切分器 |
| 诊断 | `DependencyDiagnoser` | 生成 block-to-block 图、状态和结构化 gap |
| 改写 | `ConstrainedRewriter` | 只返回 patch，不返回整条 CoT |
| 验证 | `CombinedVerifier` | 规则先行，规则通过后再调用 LLM Verifier |

`--mock` 会替换 LLM 与 analyzer，供无 GPU 的端到端测试使用。真实运行由现有
Qwen2.5-32B vLLM 服务承担评分、构图、诊断、改写和验证。

## 3. 标准记录与选择层

输入首先由 `normalize_record()` 规范为包含 `sample_id`、`question`、
`ground_truth`、`candidates` 的内部记录。选择层的统一输出包括：

```json
{
  "sample_id": "...",
  "candidate_count": 1,
  "selection_mode": "identity",
  "selected_index": 0,
  "selected_cot": "...",
  "score_components": {},
  "global_score": 0.0,
  "quality_route": "high"
}
```

单候选样本不会调用 pairwise judge；仍会经过全局评分和聚类，因此可以与跨问题
样本一起排序。多候选样本保留现有 pairwise 选择行为。

`PipelineV2._route_quality()` 将聚类结果拆为 high 与 low：

- **high**：用 `LLMAnswerJudge` 比较 GT 与最终答案区答案；`equivalent` 才可
  直接 accepted。`not_equivalent` 和 `uncertain` 写入 `answer_audit.jsonl`，不被
  降级为可改写低质量样本。
- **low**：进入 Compress、诊断和闭环验证。

答案解析位于 `answers.py`。它识别单值、答案集合、区间、选择题等，并在 CoT 中
优先取最终答案区域，避免把中间 `\boxed{}` 推导式误认为最终答案。

## 4. 严格分块与 Compress V4.2 适配

`V42Analyzer` 仍调用现有 `compress.pipeline.joint_label_v42.process_one()`，但有两项
V2 约束：

1. `atomic_segmenter.segment_atomic_v2` 使用词边界，避免把 `distributing` 中的
   `So` 等字符串误切；它也不再跨句合并长公式段。
2. refine-block prompt 规定：一个 atomic segment 一旦作为 `internal_split` 的
   parent，就不能同时出现在 `segment_group`；同 parent 的 internal split 必须是
   不重叠的精确子串。

适配器只在调用期间、受锁保护地替换 V4.2 模块全局 segmenter，调用结束后恢复，
因此不会改变 Compress V4.2 基线脚本的行为。

## 5. 依赖诊断

V4.2 原有 `dependency_graph` 保持 **block -> target** 语义。V2 诊断额外输出真正
用于逻辑闭合判断的 **block -> block** 图：

```json
{
  "block_dependencies": [
    {"from_block": 1, "to_block": 3, "relation": "derives"}
  ],
  "dependency_graph": [{"block": 3, "target": 0}],
  "dependency_state": "open",
  "gaps": [
    {
      "gap_id": "g0",
      "gap_type": "missing_derivation",
      "upstream_blocks": [1],
      "downstream_blocks": [3],
      "target_ids": [0],
      "missing_claim": "...",
      "suggested_action": "insert_patch",
      "confidence": 0.9
    }
  ]
}
```

诊断的状态为 `closed`、`open` 或 `uncertain`。`diagnosis.py` 同时进行规则回填：

- target coverage、开放依赖、只有结论、proof-gap flag、污染/不清晰 target、
  dependency cycle 都会被转成可审计的 gap；
- `contradiction`、`contaminated`、`uncertain` 等不可可靠定位的情况直接 reject；
- LLM 输出必须满足 action、anchor、target、missing claim、confidence 的固定
  schema。首次不合法时，将原始响应和校验错误反馈给模型重试一次；两次失败写为
  `diagnosis_invalid_after_retry -> uncertain`。

每次诊断尝试均保存 `raw_response` 与 `validation_errors`，便于审计模型的 schema
遵循情况。

## 6. 受限 Rewrite 与重新构图

`ConstrainedRewriter` 只能返回以下结构：

```json
{
  "rewrite_mode": "insert_patch",
  "edits": [
    {
      "after_block": 1,
      "before_block": 3,
      "replace_blocks": [],
      "new_text": "...",
      "reason": "..."
    }
  ]
}
```

允许模式为 `insert_patch`、`rewrite_span`、`format_fix`。`sanitize_rewrite()` 校验
编辑范围和诊断动作一致；`apply_rewrite()` 仅重组 supporting blocks，并只允许在
retained block 上落锚。若 `before_block` 已被裁剪而 `after_block` 仍被保留，会回退
到有效的 `after_block`；两边都不可用会明确失败，不会记录虚假的 applied edit。

Compress 默认分析 `<think>` 区域，而最终 `\boxed{}` 往往位于其外。为避免裁剪或
patch 后丢失最终输出，`PipelineV2._preserve_required_answer_suffix()` 会在候选结果
缺少 boxed 答案时，补回原始 CoT 的最终答案后缀。该步骤不修改答案内容。

每次 rewrite 后都会重新执行 V4.2 分块、标注与图构建，绝不复用旧图。达到
`max_rewrite_rounds`（当前默认 1）或不可修复后进入 rejected。

## 7. Verifier

`CombinedVerifier` 分两层运行：

1. **确定性验证**（`deterministic_verify`）检查：
   - 输出非空、最终答案与参考答案一致；
   - 需要 boxed 时格式未丢失；
   - rewrite scope 合法；
   - `dependency_state == closed`、target coverage 为 1、`dependency_open == 0`；
   - 无 unresolved gap、proof-gap baseline flag 或明显任务污染。
2. **LLM Verifier**：只有确定性验证全部通过才调用。它返回固定 JSON：

```json
{
  "verdict": "pass",
  "checks": {
    "answer": "pass",
    "format": "pass",
    "dependency": "pass",
    "faithfulness": "pass",
    "redundancy": "pass"
  },
  "failure_codes": [],
  "repairable": false
}
```

LLM prompt 明确要求全部五个 checks；解析结果保留原始 `raw_response`。只有规则
通过、LLM `verdict == pass` 且五项 checks 都为 pass 才 accepted。

## 8. `PipelineV2` 状态机

编排代码集中在 `orchestrator.py`：

```text
high
  -> semantic answer gate pass -> high_direct accepted
  -> answer mismatch / uncertain -> answer_audit + rejected

low
  -> analyze + diagnose uncertain/unrepairable -> rejected
  -> diagnose open/repairable -> rewrite once -> re-analyze -> re-diagnose -> verify
  -> diagnose closed -> supporting-block compression -> re-analyze -> re-diagnose -> verify
  -> verify pass -> low_refined accepted
  -> verify fail and no rewrite has been used and repairable -> rewrite once
  -> otherwise -> rejected
```

`lineage.jsonl` 记录 `high_selected` / `low_selected`、`diagnosed`、`rewritten`、
`verified` 与终态的时间和耗时。输出以 `sample_id + config_fingerprint` 去重，
`--resume` 不会重复追加完成记录。

## 9. 阶段产物

每个运行目录使用以下固定命名：

```text
normalized.jsonl
selected.jsonl
high_quality.jsonl
low_quality.jsonl
answer_audit.jsonl
diagnosed.jsonl
rewritten.jsonl
accepted.jsonl
rejected.jsonl
lineage.jsonl
stats.json
```

`stats.json` 聚合 selection mode、gap type、rewrite action、accept route、
verifier failure code 与答案审计统计。

## 10. 验证现状（V2.1.5）

测试入口：

```bash
source /mdr5/guest/users/zhouyan/share/cyh/envs/cot_opt/bin/activate
python -m unittest discover -s pipeline_v2/tests -v
```

当前单元测试覆盖选择路由、严格切分、诊断 retry、dependency cycle、patch scope、
think-only 答案后缀保真、Verifier 原始响应保留和 resume 幂等性。

真实 targeted smoke 位于：

```text
results/pipeline_v2/smoke_rewrite_tp1_32b_v215_20260727_6
```

其中 6 条真实题目派生的局部 proof-gap 样本产生 5 条 patch；4 条走完
`low_refined -> deterministic+llm pass`，并保存了 Verifier `raw_response`；另有
1 条在补丁后依赖仍开放，按 fail-safe 策略 rejected。对同一配置执行 `--resume` 后，
所有 JSONL 哈希保持不变。

## 11. 常用运行命令

轻量 score smoke：

```bash
python -m pipeline_v2.run \
  --input input.jsonl \
  --output-dir results/pipeline_v2/smoke_name \
  --base-url http://127.0.0.1:8000/v1 \
  --model /ky200t/models/Qwen/Qwen2.5-32B-Instruct \
  --clusterer score \
  --k-ratio 0.25 \
  --max-rewrite-rounds 1 \
  --resume
```

真实全量排序时，将 `--clusterer` 改为 `ifd_kmeans`，并按 `--gpu-id` 预留 embedding
模型使用的显存。运行结束后应清理本次启动的 vLLM/tmux 服务；不要操作共享账户下
其他人的 GPU 进程。
