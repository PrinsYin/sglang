import os
import random
import unittest
import torch
from huggingface_hub import hf_hub_download
import pickle
from typing import List, Dict, Any, Tuple
import json
import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

def load_ground_truth(
    repo_id: str = "datasets/font-info/logprobs",
    pkl_filename: str = "ground_truth.pkl",
) -> List[Dict[str, Any]]:
    pkl_path = hf_hub_download(repo_id=repo_id, filename=pkl_filename,repo_type="dataset")
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    return records

def _pack_topk(indices: List[int], logprobs: List[float]) -> Dict[int, float]:
    
    return {int(i): float(lp) for i, lp in zip(indices, logprobs)}

def _extract_srt_topk(entry) -> Dict[int, float]:
    
    return {int(tok_id): float(lp) for lp, tok_id, _ in entry}

def _allclose_dict(
    a: Dict[int, float],
    b: Dict[int, float],
    rtol: float = 1e-1,
    atol: float = 1e-6,
    require_same_keys: bool = True,
    position: int = -1,
) -> Tuple[bool, str]:
    missing = set(a) - set(b)
    extra = set(b) - set(a)
    if len(missing) + len(extra) > 20:
        return False, f"top-k token 差异过大; missing={sorted(missing)} extra={sorted(extra)}"

    common = set(a) & set(b)
    if not common:
        return False, "没有共同 token 可比较"
    diffs = []
    for tid in common:
        diffs.append(abs(a[tid] - b[tid]))
    max_diff = max(diffs)
    ok = all(abs(a[tid] - b[tid]) <= (atol + rtol * max(abs(a[tid]), abs(b[tid]))) for tid in common)
    
    if not ok:
        return False, f"logprob 偏差过大={max_diff:.6g} (atol={atol}, rtol={rtol})"
    print(f"position={position} top-k token; missing={sorted(missing)} extra={sorted(extra)} logprob 最大偏差={max_diff:.6g} ")
    return True, ""

class TestChunkedLogprobsAgainstHFStored(CustomTestCase):
    def test_against_hf_stored_topk(self):
        records = load_ground_truth(repo_id="font-info/logprobs", pkl_filename="ground_truth.pkl")
        assert len(records) > 0, "ground_truth 为空；确认 HF 仓库与文件名是否正确"

        # rng = random.Random(1234)
        subset = random.sample(records, k=min(len(records)/2, len(records)))
        print(f"testing on {len(subset)} samples")

        os.environ["SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"] = "1"  
        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
            mem_fraction_static=0.6,
        )

        try:
            for rec in subset:
                with open("rec.json", "w") as f:
                    json.dump(rec, f, indent=2)
                ids: List[int] = rec["ids"]
                start_pos: int = rec["start_pos"]
                gt_inputs = rec["input_topk_logprobs"]  
                gt_next = rec["first_output_topk_logprobs"]

                
                top_k = len(gt_next["topk_indices"])
                sampling_params = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": max(top_k, 1),   
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

                
                
                srt_input_top = meta["input_top_logprobs"][1:]
                with open("srt_input_top.json", "w") as f:
                    json.dump(srt_input_top, f, indent=2)
                
                
                # print(f"srt_input_top: {len(srt_input_top)}")
                # print(f"start_pos: {start_pos}")
                # print(f"ids: {len(rec['input_topk_logprobs'])}")
                assert len(srt_input_top) == len(gt_inputs), f"input_top 长度不一致（SGLang vs ground_truth）, srt_input_top: {len(srt_input_top)}, gt_inputs: {len(gt_inputs)}"

                
                for srt_entry, gt_entry in zip(srt_input_top, gt_inputs):
                    position = gt_entry["position"]
                    if not srt_entry:
                        print(f"srt_entry: {srt_entry}")
                        continue
                    srt_map = _extract_srt_topk(srt_entry)
                    gt_map = _pack_topk(gt_entry["topk_indices"], gt_entry["topk_logprobs"])

                    ok, msg = _allclose_dict(srt_map, gt_map, rtol=0.2, atol=1e-6, require_same_keys=True,position=position)
                    self.assertTrue(ok, f"[input pos={gt_entry['position']}] {msg}")

                
                
                assert len(meta["output_top_logprobs"]) >= 1, "没有拿到输出 top-k"
                srt_next_map = _extract_srt_topk(meta["output_top_logprobs"][0])
                gt_next_map = _pack_topk(gt_next["topk_indices"], gt_next["topk_logprobs"])
                ok, msg = _allclose_dict(srt_next_map, gt_next_map, rtol=0.2, atol=1e-6, require_same_keys=True)
                self.assertTrue(ok, f"[first_output pos={gt_next['position']}] {msg}")

        finally:
            engine.shutdown()
            del engine
            torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main()
