import os, pickle, numpy as np
import torch
import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
import random
import unittest
import requests
import io
import time

MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
INPUT_PKL_URL = "https://huggingface.co/datasets/font-info/logprobs/resolve/main/sglang_baseline.pkl"
TOP_K = 20
BATCH_SIZE = 50
NUM_SAMPLES = 500
MAX_RETRIES = 3
RETRY_DELAY = 2
TOLERANCE_MAX_DIFF = 1.5
TOLERANCE_MEAN_DIFF = 0.1

os.environ["RETURN_ORIGINAL_LOGPROB"] = "True"


class TestLogprobs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang Engine with {MODEL_NAME}...")
        cls.engine = sgl.Engine(
            model_path=MODEL_NAME,
            random_seed=42,
            skip_tokenizer_init=True,
            mem_fraction_static=0.6,
            max_running_requests=1,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - shutdown the engine."""
        cls.engine.shutdown()
        torch.cuda.empty_cache()

    def load_test_data(self):
        """Load test data from Hugging Face dataset with retry mechanism."""
        print(f"Loading data from {INPUT_PKL_URL}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(INPUT_PKL_URL, timeout=30)
                response.raise_for_status()
                
                with io.BytesIO(response.content) as f:
                    records = pickle.load(f)
                
                if not records:
                    raise ValueError("Empty dataset")
                
                print(f"Successfully loaded {len(records)} records")
                return records
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Failed to load data after {MAX_RETRIES} attempts: {e}")
                time.sleep(RETRY_DELAY)

    def compare_meta(self, metaA, metaB):
        """Compare metadata between two outputs and return max and mean differences."""
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

    def test_logprobs_comparison(self):
        """Test the logprobs comparison functionality."""
        # Load test data with retry mechanism
        records = self.load_test_data()
        records = random.sample(records, k=min(NUM_SAMPLES, len(records)))
        random.shuffle(records)
        print(f"Testing with {len(records)} samples")

        all_max, all_mean = [], []
        
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            input_ids = [rec["ids"] for rec in batch]
            logprob_start_lens = [rec["start_pos"] for rec in batch]

            # Sampling param per request
            sampling_params = [ 
                {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": TOP_K,
                    "max_new_tokens": 1
                } for _ in batch
            ]

            outputs = self.engine.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=True,
                logprob_start_len=logprob_start_lens,
                top_logprobs_num=TOP_K,
            )

            for rec, output in zip(batch, outputs):
                metaA = rec["meta"]
                metaB = output["meta_info"]

                max_diff, mean_diff = self.compare_meta(metaA, metaB)
                all_max.append(max_diff)
                all_mean.append(mean_diff)

                # print(f"[Sample {rec['id']}] max Δ={max_diff:.6g}, mean Δ={mean_diff:.6g}")

        print("\n=== Overall statistics ===")
        max_of_max = max(all_max)
        mean_of_mean = np.mean(all_mean)
        print(f"max of max Δ={max_of_max:.6g}")
        print(f"mean of mean Δ={mean_of_mean:.6g}")

        # Basic validation
        self.assertIsInstance(all_max, list)
        self.assertIsInstance(all_mean, list)
        self.assertGreater(len(all_max), 0, "No test samples processed")
        
        # Tolerance checks with clear error messages
        failed_samples = []
        for i, (max_diff, mean_diff) in enumerate(zip(all_max, all_mean)):
            if max_diff > TOLERANCE_MAX_DIFF:
                failed_samples.append(f"Sample {i}: max_diff={max_diff:.6g} > {TOLERANCE_MAX_DIFF}")
            if mean_diff > TOLERANCE_MEAN_DIFF:
                failed_samples.append(f"Sample {i}: mean_diff={mean_diff:.6g} > {TOLERANCE_MEAN_DIFF}")
        
        if failed_samples:
            self.fail(f"Tolerance exceeded in {len(failed_samples)} samples:\n" + "\n".join(failed_samples[:10]))
        
        print(f"✅ All {len(all_max)} samples passed tolerance checks (max_diff ≤ {TOLERANCE_MAX_DIFF}, mean_diff ≤ {TOLERANCE_MEAN_DIFF})")


if __name__ == "__main__":
    unittest.main()