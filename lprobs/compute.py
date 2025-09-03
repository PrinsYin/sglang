import json
import pickle
import requests
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# -----------------------------
# Config
# -----------------------------
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
NUM_SAMPLES = 500
NUM_META_SAMPLES = 2
TOP_K = 50
OUTPUT_PKL = "ground_truth.pkl"
OUTPUT_META = "ground_truth_meta.json"

# -----------------------------
# Load dataset
# -----------------------------
print("Downloading ShareGPT dataset...")
data = json.loads(requests.get(SHAREGPT_URL).text)
with open("sharegpt.json", "w") as f:
    json.dump(data[:10], f, ensure_ascii=False)
data = data[:NUM_SAMPLES]

texts = [
    s["conversations"][0]["value"]
    for s in data
    if "conversations" in s and len(s["conversations"]) > 0
]
# -----------------------------
# Load HF model
# -----------------------------
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
)
model.eval()

# -----------------------------
# Helper function
# -----------------------------
def compute_topk_logprobs(ids, top_k=10, start_pos=0):
    """
    给定 token ids，返回从 start_pos 开始的 top-k logprobs。
    返回：
    - input_topk_logprobs: 每个 input 位置的 top-k 分布
    - first_output_topk_logprobs: 最后一个 input 的下一个 token 分布
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
            "token_id": ids[i],
            "topk_indices": topk_inds.tolist(),
            "topk_logprobs": topk_vals.tolist()
        })

    topk_vals, topk_inds = torch.topk(log_probs[len(ids) - 1], top_k)
    first_output_topk_logprobs = {
        "position": len(ids) - 1,
        "token_id": ids[-1],
        "topk_indices": topk_inds.tolist(),
        "topk_logprobs": topk_vals.tolist()
    }

    return input_topk_logprobs, first_output_topk_logprobs

# -----------------------------
# Main loop
# -----------------------------
records = []

print(f"Computing top-{TOP_K} logprobs for {NUM_SAMPLES} samples...")
for i, text in enumerate(texts):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 5:  # 至少有空间留出 output
        continue

    start_pos = np.random.randint(0, len(ids) - 3)

    input_topk, first_output_topk = compute_topk_logprobs(ids, top_k=TOP_K, start_pos=start_pos)

    rec = {
        "id": i,
        "text": text,
        "ids": ids,
        "start_pos": start_pos,
        "input_topk_logprobs": input_topk,
        "first_output_topk_logprobs": first_output_topk
    }
    records.append(rec)

    if (i + 1) % 5 == 0:
        print(f"Processed {i+1}/{NUM_SAMPLES}")

# -----------------------------
# Save
# -----------------------------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(records, f)

with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(records[:NUM_META_SAMPLES], f, ensure_ascii=False, indent=2)

print(f"✅ Done! Saved {len(records)} full samples to {OUTPUT_PKL}")
print(f"   Saved {NUM_META_SAMPLES} meta samples to {OUTPUT_META}")