"""
Test original log probability alignment between SGLang and Hugging Face.

Changed: PROMPTS are now loaded from ShareGPT dataset at SHAREGPT_URL.
Everything else remains the same.
"""

import os
import io
import json
import random
import unittest
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# ------------------------- Configurable via env ------------------------- #
MODEL_ID = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# ShareGPT source & prompt sampling controls
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
# Number of prompts to sample (100~200 recommended); default 150.
SHAREGPT_N = int(os.getenv("SHAREGPT_N", "150"))
# Optional prompt length guards
PROMPT_MIN_CHARS = int(os.getenv("PROMPT_MIN_CHARS", "8"))
PROMPT_MAX_CHARS = int(os.getenv("PROMPT_MAX_CHARS", "4000"))
# Local cache path (optional)
SHAREGPT_CACHE = os.getenv("SHAREGPT_CACHE", "sharegpt_split.json")

TOP_LOGPROBS_NUM = 100
NUM_RANDOM_TOKEN_IDS = 500
RTOL = 0.20
ATOL = 0.00
# ------------------------------------------------

torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ------------------------- Prompt loading ------------------------- #
_DEF_FALLBACK_PROMPTS = [
    "Hello, my name is",
    "The future of AI is",
    (
        "The president of the United States is "
        "The president of the United States isThe president of the United States is"
        "The president of the United States isThe president of the United States is"
        "The president of the United States isThe president of the United States is"
        "The president of the United States isThe president of the United States is"
        "The president of the United States isThe president of the United States is"
        "The president of the United States isThe president of the United States is"
        "The president of the United States is"
    ),
    "The capital of France is ",
]


def _first_user_utterance(conv_item: dict) -> str:
    """
    Try to extract the first 'human'/'user' message from a ShareGPT conversation item.
    Handles a few common schema variants.
    """
    conv = conv_item.get("conversations") or conv_item.get("conversation") or []
    for turn in conv:
        # Common keys in this dataset
        speaker = turn.get("from") or turn.get("role") or turn.get("speaker")
        text = turn.get("value") or turn.get("content") or turn.get("text")
        if not isinstance(text, str):
            continue
        if speaker is None:
            # Fallback heuristic: some entries might just alternate; assume first is user
            return text.strip()
        if str(speaker).lower() in {"human", "user"}:
            return text.strip()
    return ""


def load_prompts_from_sharegpt(
    url: str,
    n: int,
    min_chars: int,
    max_chars: int,
    cache_path: str = None,
    seed: int = 1234,
) -> List[str]:
    assert n > 0, "n must be positive"
    random.seed(seed)

    data = None

    # 1) Try cache if provided
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

    # 2) If no cache, download
    if data is None:
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            # Keep a local cache if requested
            data = resp.json()
            if cache_path:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Failed to download ShareGPT dataset: {e}")
            return _DEF_FALLBACK_PROMPTS

    # 3) Extract prompts
    prompts = []
    seen = set()
    for item in data:
        p = _first_user_utterance(item)
        if not p:
            continue
        p = " ".join(p.split())  # normalize whitespace
        if not (min_chars <= len(p) <= max_chars):
            continue
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)

    if not prompts:
        print("[WARN] No valid prompts extracted; falling back to defaults.")
        return _DEF_FALLBACK_PROMPTS

    random.shuffle(prompts)
    n = max(1, min(n, len(prompts)))
    return prompts[:n]


class TestOriginalLogprob(unittest.TestCase):
    def setUp(self):
        # ----- HF side (float32 weights) -----
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="auto"
        )

        # Shared sampling parameters
        self.sampling_params = {
            "temperature": 0.5,  # SGLang uses 0.5, but original logprobs are used 1.0
            "top_p": 1.0,
            "top_k": 10,
            "max_new_tokens": 1,
        }

        # ----- Load prompts from ShareGPT (fallback to defaults if needed) -----
        try:
            self.prompts = load_prompts_from_sharegpt(
                url=SHAREGPT_URL,
                n=SHAREGPT_N,
                min_chars=PROMPT_MIN_CHARS,
                max_chars=PROMPT_MAX_CHARS,
                cache_path=SHAREGPT_CACHE,
                seed=1234,
            )
            print(f"[INFO] Loaded {len(self.prompts)} prompts from ShareGPT.")
        except Exception as e:
            print(f"[WARN] Exception during prompt loading: {e}")
            self.prompts = _DEF_FALLBACK_PROMPTS

    # ---------------------------------------------------------------------
    # Helper: compare one SGLang block (token_logprobs / top_logprobs / ids_logprobs)
    #         against a reference HF log-prob vector.
    # ---------------------------------------------------------------------
    def assert_logprobs_block_equal(
        self,
        hf_log_probs: torch.Tensor,  # [V]
        token_log_probs: list,
        top_log_probs: list,
        ids_log_probs: list,
        random_token_ids: list,
        tag: str = "",
    ):
        vals, idxs, _ = zip(*token_log_probs)
        sgl_vals = torch.tensor(vals, device=self.hf_model.device, dtype=torch.float32)
        sgl_idxs = torch.tensor(idxs, device=self.hf_model.device, dtype=torch.long)
        hf_vals = hf_log_probs[sgl_idxs]

        self.assertTrue(
            torch.allclose(hf_vals, sgl_vals, rtol=RTOL, atol=ATOL),
            msg=f"[{tag}] token-level mismatch at indices {sgl_idxs.tolist()}",
            )

        hf_topk, _ = torch.topk(hf_log_probs, k=TOP_LOGPROBS_NUM, dim=-1)

        sgl_topk = torch.tensor(
            [float(t[0]) for t in top_log_probs[0] if t and t[0] is not None][
                :TOP_LOGPROBS_NUM
            ],
            dtype=torch.float32,
            device=self.hf_model.device,
        )

        k = min(hf_topk.numel(), sgl_topk.numel())
        self.assertTrue(
            torch.allclose(hf_topk[:k], sgl_topk[:k], rtol=RTOL, atol=ATOL),
            msg=f"[{tag}] top-k mismatch",
        )

        indices = torch.tensor(
            random_token_ids, dtype=torch.long, device=hf_log_probs.device
        )

        hf_token_ids = hf_log_probs[indices]

        sgl_token_ids = torch.tensor(
            [v for v, _, _ in ids_log_probs[0]],
            device=self.hf_model.device,
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(hf_token_ids, sgl_token_ids, rtol=RTOL, atol=ATOL),
            msg=f"[{tag}] token-IDs mismatch",
        )

        # Optional: print max abs diff for quick diagnostics
        max_diff = torch.max(torch.abs(hf_vals - sgl_vals)).item()
        print(f"[{tag}] max|diff| token-level = {max_diff:.4f}")

    def test_logprob_match(self):
        vocab_size = self.tokenizer.vocab_size

        for env_val in ["True", "False"]:
            with self.subTest(return_original_logprob=env_val):
                os.environ["RETURN_ORIGINAL_LOGPROB"] = env_val

                # ----- SGLang side -----
                sgl_engine = sgl.Engine(
                    model_path=MODEL_ID,
                    skip_tokenizer_init=True,
                    trust_remote_code=True,
                    mem_fraction_static=0.60,
                )

                for prompt in self.prompts:
                    random_token_ids = sorted(
                        random.sample(range(vocab_size), NUM_RANDOM_TOKEN_IDS)
                    )

                    enc = self.tokenizer(prompt, return_tensors="pt")
                    input_ids = enc["input_ids"].to(self.hf_model.device)
                    attn_mask = enc["attention_mask"].to(self.hf_model.device)

                    with torch.inference_mode():
                        hf_out = self.hf_model(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            return_dict=True,
                        )
                    logits = hf_out.logits[:, -1, :]  # [1, V]
                    hf_log_probs = F.log_softmax(
                        logits.float() / self.sampling_params["temperature"], dim=-1
                    )[0]
                    hf_original_log_probs = F.log_softmax(logits.float(), dim=-1)[0]

                    outputs = sgl_engine.generate(
                        input_ids=input_ids[0].tolist(),
                        sampling_params=self.sampling_params,
                        return_logprob=True,
                        top_logprobs_num=TOP_LOGPROBS_NUM,
                        token_ids_logprob=random_token_ids,
                    )

                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    meta = outputs["meta_info"]

                    # Check original logprobs only if enabled
                    if env_val.lower() == "true":
                        self.assert_logprobs_block_equal(
                            hf_log_probs=hf_original_log_probs,
                            token_log_probs=meta["output_token_logprobs"],
                            top_log_probs=meta["output_top_logprobs"],
                            ids_log_probs=meta["output_token_ids_logprobs"],
                            random_token_ids=random_token_ids,
                            tag=f"Original logprobs SGLang vs HF: (env={env_val})",
                        )
                    else:
                        # Always check regular logprobs
                        self.assert_logprobs_block_equal(
                            hf_log_probs=hf_log_probs,
                            token_log_probs=meta["output_token_logprobs"],
                            top_log_probs=meta["output_top_logprobs"],
                            ids_log_probs=meta["output_token_ids_logprobs"],
                            random_token_ids=random_token_ids,
                            tag=f"logprobs SGLang vs HF: (env={env_val})",
                        )
                sgl_engine.shutdown()


if __name__ == "__main__":
    unittest.main()