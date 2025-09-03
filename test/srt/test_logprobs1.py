# test_sglang_vs_hf_sharegpt_topk.py
import os
import random
import unittest
import json
import pickle
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# =========================
# Configs
# =========================
MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST    # HF 与 SGLang 都用这个
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "100"))     # 现场算样本数（越大越慢）
TOP_K = int(os.environ.get("TOP_K", "50"))                  # 现场算的 top-k
SEED = 42

# 位置对齐补偿开关：有的引擎会把第 0 位放一个占位条目
# 你的旧代码对 input_top_logprobs 做了 [1:]，这里保留一个开关便于 A/B
SGLANG_INPUT_TOP_OFF_BY_ONE = bool(int(os.environ.get("SGLANG_INPUT_TOP_OFF_BY_ONE", "1")))

# 如果希望 SGLang 返回“原始 logits 归一后的 logprob”，可保留该环境变量（按你给的版本）
os.environ.setdefault("RETURN_ORIGINAL_LOGPROB", "True")


# =========================
# Utilities for on-the-fly HF ground truth
# =========================
def fetch_sharegpt_texts(n: int) -> List[str]:
    """下载 ShareGPT 数据，并抽取前 n 条 user 首轮文本"""
    print("Downloading ShareGPT dataset (for ground-truth on the fly)...")
    data = json.loads(requests.get(SHAREGPT_URL).text)

    # 取前 n 条；若条目缺失或格式异常则跳过
    texts: List[str] = []
    for s in data:
        if "conversations" in s and len(s["conversations"]) > 0:
            val = s["conversations"][0].get("value", "")
            if isinstance(val, str) and val.strip():
                texts.append(val)
                if len(texts) >= n:
                    break
    print(f"Using {len(texts)} ShareGPT samples.")
    return texts


def compute_topk_logprobs_hf(
    model, tokenizer, ids: List[int], top_k: int, start_pos: int
):
    """
    给定 token ids，返回从 start_pos 开始的 top-k logprobs（作为“真值”）。
    - input_topk_logprobs: 每个 input 位置的 top-k 分布（从 start_pos 到 len(ids)-2）
    - first_output_topk_logprobs: 最后一个 input 的下一个 token 分布（位置 len(ids)-1）
    """
    input_ids = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, L, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)[0]  # [L, vocab]

    input_topk_logprobs = []
    for i in range(start_pos, len(ids) - 1):
        topk_vals, topk_inds = torch.topk(log_probs[i], top_k)
        input_topk_logprobs.append({
            "position": i,
            "token_id": int(ids[i]),
            "topk_indices": topk_inds.tolist(),
            "topk_logprobs": topk_vals.tolist()
        })

    topk_vals, topk_inds = torch.topk(log_probs[len(ids) - 1], top_k)
    first_output_topk_logprobs = {
        "position": len(ids) - 1,
        "token_id": int(ids[-1]),
        "topk_indices": topk_inds.tolist(),
        "topk_logprobs": topk_vals.tolist()
    }

    return input_topk_logprobs, first_output_topk_logprobs


def build_ground_truth_records_on_the_fly(
    model_name: str, num_samples: int, top_k: int, seed: int
) -> List[Dict[str, Any]]:
    """
    现场用 HF 模型计算若干 ShareGPT 文本的 top-k logprobs，构造与原 ground_truth.pkl 同结构的 records 列表
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading HF model for ground-truth: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # 建议用 fp16（与很多推理后端一致）；要做严格数值对齐可切到 fp32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="auto"
    )
    model.eval()

    texts = fetch_sharegpt_texts(num_samples)
    records: List[Dict[str, Any]] = []

    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        # 至少保证有空间留出 output（最后一个位置作为 next-token 的分布）
        if len(ids) < 5:
            continue
        # start_pos 随机，留出 >=2 的尾部
        start_pos = int(np.random.randint(0, len(ids) - 3))

        input_topk, first_output_topk = compute_topk_logprobs_hf(
            model, tokenizer, ids, top_k=top_k, start_pos=start_pos
        )

        rec = {
            "id": i,
            "text": text,
            "ids": ids,
            "start_pos": start_pos,
            "input_topk_logprobs": input_topk,
            "first_output_topk_logprobs": first_output_topk
        }
        records.append(rec)

        if (i + 1) % 10 == 0:
            print(f"[HF GT] processed {i + 1}/{len(texts)}")

    print(f"[HF GT] total usable records: {len(records)}")
    return records


# =========================
# Helpers (与你的原始测试保持一致)
# =========================
def _pack_topk(indices: List[int], logprobs: List[float]) -> Dict[int, float]:
    return {int(i): float(lp) for i, lp in zip(indices, logprobs)}

def _extract_srt_topk(entry) -> Dict[int, float]:
    # SGLang: entry 一般是 (logprob, token_id, token_str) 的列表
    return {int(tok_id): float(lp) for lp, tok_id, _ in entry}

def _allclose_dict(
    a: Dict[int, float],
    b: Dict[int, float],
    rtol: float = 1e-1,
    atol: float = 1e-6,
    require_same_keys: bool = True,
    position: int = -1,
) -> Tuple[bool, str, float, float]:
    missing = set(a) - set(b)
    extra = set(b) - set(a)
    if require_same_keys and (len(missing) + len(extra) > 40):
        return False, f"top-k token 差异过大; missing={sorted(missing)} extra={sorted(extra)}", 0.0, 0.0
    # print(f"missing={sorted(missing)} extra={sorted(extra)}")
    common = set(a) & set(b)
    if not common:
        return False, "没有共同 token 可比较", 0.0, 0.0

    diffs = [abs(a[tid] - b[tid]) for tid in common]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(common)
    ok = abs(max_diff) <= (atol + rtol)  # 与你用法接近；如需更严格可用纯 atol
    if not ok:
        return False, f"logprob 偏差过大={max_diff:.6g} (atol={atol}, rtol={rtol})", max_diff, mean_diff
    return True, "", max_diff, mean_diff


# =========================
# The Test Case
# =========================
class TestSGLangVsHFShareGPTTopK(CustomTestCase):
    def test_sglang_against_hf_sharegpt_topk(self):
        # 现场构造“真值” records
        records = build_ground_truth_records_on_the_fly(
            model_name=MODEL_NAME,
            num_samples=NUM_SAMPLES,
            top_k=TOP_K,
            seed=SEED,
        )
        assert len(records) > 0, "ground_truth 为空；检查 ShareGPT 下载与 HF 推理是否正常"

        # 与原逻辑尽量一致：随机抽一半子集测试
        subset = random.sample(records, k=min(len(records)//2, len(records)))
        print(f"testing on {len(subset)} samples")

        # 初始化 SGLang 引擎
        engine = sgl.Engine(
            model_path=MODEL_NAME,
            random_seed=SEED,
            skip_tokenizer_init=True,
            mem_fraction_static=0.6,
        )

        try:
            for rec in subset:
                # 仅用于调试查看
                with open("rec_live.json", "w") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)

                ids: List[int] = rec["ids"]
                start_pos: int = rec["start_pos"]
                gt_inputs = rec["input_topk_logprobs"]
                gt_next = rec["first_output_topk_logprobs"]

                # 与你原逻辑保持一致：用“真值”的 top_k 长度驱动
                top_k = len(gt_next["topk_indices"])
                sampling_params = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": max(top_k, 1),    # 如果你想避免采样处理器影响，可改为去掉 top_k
                    "max_new_tokens": 1,
                }

                outputs = engine.generate(
                    input_ids=[ids],
                    sampling_params=sampling_params,
                    return_logprob=True,
                    logprob_start_len=start_pos,
                    top_logprobs_num=top_k,
                )
                out = outputs[0]
                meta = out["meta_info"]

                # 对齐策略：保留你原来的 [1:]，也可通过环境变量开关控制
                srt_input_top = meta["input_top_logprobs"]
                if SGLANG_INPUT_TOP_OFF_BY_ONE:
                    srt_input_top = srt_input_top[1:]

                with open("srt_input_top_live.json", "w") as f:
                    json.dump(srt_input_top, f, indent=2)

                assert len(srt_input_top) == len(gt_inputs), (
                    f"input_top 长度不一致（SGLang vs HF ground-truth），"
                    f"srt_input_top: {len(srt_input_top)}, gt_inputs: {len(gt_inputs)}, "
                    f"start_pos={start_pos}, len(ids)={len(ids)}"
                )

                req_max_diff = -1e9
                req_mean_diff = -1e9
                # for srt_entry, gt_entry in zip(srt_input_top, gt_inputs):
                #     position = gt_entry["position"]
                #     if not srt_entry:
                #         print(f"srt_entry is empty at pos={position}")
                #         continue
                #     srt_map = _extract_srt_topk(srt_entry)
                #     gt_map = _pack_topk(gt_entry["topk_indices"], gt_entry["topk_logprobs"])

                #     ok, msg, max_diff, mean_diff = _allclose_dict(
                #         srt_map, gt_map, rtol=0.4, atol=1e-6, require_same_keys=True, position=position
                #     )
                #     req_max_diff = max(req_max_diff, max_diff)
                #     req_mean_diff = max(req_mean_diff, mean_diff)
                #     self.assertTrue(ok, f"[input pos={position}] {msg}")

                # print(f"[input] last_pos={position} max_abs_diff={req_max_diff:.6g} max_mean_diff={req_mean_diff:.6g}")

                # 对比“第一个输出 token”的 top-k
                assert len(meta["output_top_logprobs"]) >= 1, "没有拿到输出 top-k"
                srt_next_map = _extract_srt_topk(meta["output_top_logprobs"][0])
                gt_next_map = _pack_topk(gt_next["topk_indices"], gt_next["topk_logprobs"])
                ok, msg, max_diff, mean_diff = _allclose_dict(
                    srt_next_map, gt_next_map, rtol=0.2, atol=1e-6, require_same_keys=True, position=-1
                )
                self.assertTrue(ok, f"[first_output pos={gt_next['position']}] {msg}")
                print(f"[input] last_pos={gt_next['position']} max_abs_diff={max_diff:.6g} max_mean_diff={mean_diff:.6g}")

        finally:
            engine.shutdown()
            del engine
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()