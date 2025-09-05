import os, pickle, numpy as np, random
import torch
from transformers import AutoTokenizer
import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
INPUT_PKL = "runA.pkl"

os.environ["RETURN_ORIGINAL_LOGPROB"] = "True"

def compare_meta(metaA, metaB):
    """比较两个 meta_info，返回 (max_diff, mean_diff)"""
    diffs = []
    for key in ["input_top_logprobs", "output_top_logprobs"]:
        arrA, arrB = metaA[key], metaB[key]
        for e1, e2 in zip(arrA, arrB):
            if not e1 or not e2:
                continue
            dmapA = {tid: lp for lp, tid, _ in e1}
            dmapB = {tid: lp for lp, tid, _ in e2}
            common = dmapA.keys() & dmapB.keys()
            for tid in common:
                diffs.append(abs(dmapA[tid] - dmapB[tid]))
    if not diffs:
        return 0.0, 0.0
    return max(diffs), float(np.mean(diffs))

def main():
    with open(INPUT_PKL, "rb") as f:
        records = pickle.load(f)

    print(f"Loaded {len(records)} samples from {INPUT_PKL}")

    # 打乱顺序
    random.shuffle(records)

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print(f"Launching SGLang Engine with {MODEL_NAME}...")
    engine = sgl.Engine(
        model_path=MODEL_NAME,
        random_seed=42,
        skip_tokenizer_init=True,
        mem_fraction_static=0.6,
    )

    all_max, all_mean = [], []
    try:
        for rec in records:
            ids = rec["ids"]
            start_pos = rec["start_pos"]
            metaA = rec["meta"]

            outputs = engine.generate(
                input_ids=[ids],
                sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": 50, "max_new_tokens": 1},
                return_logprob=True,
                logprob_start_len=start_pos,
                top_logprobs_num=50,
            )
            metaB = outputs[0]["meta_info"]

            max_diff, mean_diff = compare_meta(metaA, metaB)
            all_max.append(max_diff)
            all_mean.append(mean_diff)

            print(f"[Sample {rec['id']}] max Δ={max_diff:.6g}, mean Δ={mean_diff:.6g}")

        # 整体统计
        print("\n=== Overall statistics ===")
        print(f"max of max Δ={max(all_max):.6g}")
        print(f"mean of mean Δ={np.mean(all_mean):.6g}")

    finally:
        engine.shutdown()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()