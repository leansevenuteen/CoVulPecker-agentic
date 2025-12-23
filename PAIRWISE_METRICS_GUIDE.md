# MAVUL Pairwise Metrics Evaluation Guide

This guide explains how to use the MAVUL pairwise metrics evaluation system to assess your multi-agent vulnerability detection pipeline.

## Overview

The MAVUL pairwise metrics evaluate how well your system can distinguish between vulnerable code (pre-patch) and fixed code (post-patch). This is a more rigorous evaluation than simple accuracy metrics.

## Metrics Explained

### P-C (Perfect-Correct) - HIGHER IS BETTER
- **Definition**: Correctly identifies pre-patch as vulnerable AND post-patch as non-vulnerable
- **Interpretation**: The system perfectly understands both the vulnerability and the fix
- **This is the ideal outcome**

### P-V (Perfect-Vulnerable) - LOWER IS BETTER
- **Definition**: Correctly identifies pre-patch as vulnerable BUT incorrectly identifies post-patch as vulnerable
- **Interpretation**: The system finds the vulnerability but fails to recognize that the patch fixed it
- **Problem**: False positive on patched code

### P-B (Perfect-Benign) - LOWER IS BETTER
- **Definition**: Incorrectly identifies pre-patch as non-vulnerable AND correctly identifies post-patch as non-vulnerable
- **Interpretation**: The system misses the vulnerability entirely in both versions
- **Problem**: False negative on vulnerable code

### P-R (Perfect-Reverse) - LOWER IS BETTER
- **Definition**: Incorrectly identifies pre-patch as non-vulnerable AND incorrectly identifies post-patch as vulnerable
- **Interpretation**: Completely backwards understanding
- **This is the worst outcome**

## Usage

### Step 1: Run Detection on Pairwise Dataset

The modified `run_detection_on_dataset.py` now analyzes BOTH pre-patch and post-patch code for each sample:

```bash
# Run on all samples
python run_detection_on_dataset.py --dataset agentic_dataset.json --output pairwise_results.json

# Run on first 10 samples for testing (recommended for initial testing)
python run_detection_on_dataset.py --dataset agentic_dataset.json --output pairwise_results.json --limit 10

# Customize max iterations
python run_detection_on_dataset.py --dataset agentic_dataset.json --output pairwise_results.json --max_iterations 5
```

**What happens:**
1. For each sample, the pipeline runs TWICE:
   - Once on pre-patch code (vulnerable version)
   - Once on post-patch code (fixed version)
2. Results include both predictions and automatically calculated pairwise category
3. Progress is saved every 10 samples in case of interruption

**Expected output format:**
```json
{
  "sample_id": 0,
  "cwe_id": "CWE-401",
  "true_label": 1,
  "pre_patch_prediction": {
    "predicted_label": "vulnerable",
    "confidence": 0.85,
    "vulnerabilities_found": 2,
    "vulnerability_confirmed": true,
    "critic_decision": "approved"
  },
  "post_patch_prediction": {
    "predicted_label": "non-vulnerable",
    "confidence": 0.72,
    "vulnerabilities_found": 0,
    "vulnerability_confirmed": false,
    "critic_decision": "approved"
  },
  "pairwise_category": "P-C",
  "iterations": 1,
  "status": "success"
}
```

### Step 2: Evaluate Pairwise Metrics

Once you have results with both predictions, evaluate the metrics:

```bash
# Basic evaluation
python evaluate_pairwise_metrics.py pairwise_results.json

# Save evaluation report
python evaluate_pairwise_metrics.py pairwise_results.json --output evaluation_report.json

# Exclude detailed per-sample breakdown (for large datasets)
python evaluate_pairwise_metrics.py pairwise_results.json --output evaluation_report.json --no-detailed
```

**Output includes:**
1. **Console summary** with overall metrics and per-CWE breakdown
2. **JSON report** (`evaluation_report.json`) with structured metrics data
3. **Text report** (`evaluation_report.txt`) with human-readable summary

**Example output:**
```
======================================================================
MAVUL PAIRWISE METRICS EVALUATION
======================================================================
Total Pairs Analyzed: 100

P-C (Perfect-Correct):        64 ( 64.0%)  <- Higher is better
P-V (Perfect-Vulnerable):     18 ( 18.0%)  <- Lower is better
P-B (Perfect-Benign):         12 ( 12.0%)  <- Lower is better
P-R (Perfect-Reverse):         6 (  6.0%)  <- Lower is better

Pairwise Accuracy (P-C): 64.00%
======================================================================
```

## Interpreting Results

### Good Performance
- **P-C > 60%**: Strong ability to distinguish vulnerable from fixed code
- **P-V + P-R < 20%**: Low rate of serious errors
- **P-B < 15%**: Reasonable detection rate

### Areas for Improvement
- **High P-V**: System is too aggressive, needs better understanding of patches
- **High P-B**: System misses vulnerabilities, needs better detection capabilities
- **High P-R**: System logic is flawed, needs fundamental improvements

## Comparison with MAVUL Paper

According to the MAVUL paper:
- MAVUL achieved **62% higher pairwise accuracy** than existing multi-agent systems
- Performance improved with **increased communication rounds** between agents
- Your system already has multi-agent architecture (Classifier → Detector → Reasoner → Critic → Verifier)

To improve your metrics:
1. Increase `--max_iterations` to allow more critic feedback rounds
2. Enhance the reasoning agent's understanding of code context
3. Improve the verification agent's PoC validation

## Dataset Requirements

Your dataset must have:
- `pre_patch`: Code before the vulnerability was fixed
- `post_patch`: Code after the vulnerability was fixed
- `label`: Ground truth (1 = vulnerable, 0 = non-vulnerable)
- `cwe_id`: (Optional) For per-CWE breakdown

## Files

- **run_detection_on_dataset.py**: Modified to run on both code versions
- **evaluate_pairwise_metrics.py**: Calculate and display MAVUL metrics
- **agentic_dataset.json**: Your pairwise vulnerability dataset

## Troubleshooting

### "UNKNOWN: 100%" in evaluation
Your results are in old format (only pre-patch analyzed). Re-run with modified `run_detection_on_dataset.py`.

### API Rate Limits
Since the pipeline runs twice per sample, you'll make 2x API calls. Consider:
- Using `--limit` for testing
- Adding delays between samples if needed
- Using a local LLM if available

### Memory Issues
For large datasets:
- Use `--no-detailed` flag in evaluation
- Process in batches using `--limit`
- Results are saved every 10 samples

## Example Workflow

```bash
# 1. Test on small sample
python run_detection_on_dataset.py --dataset agentic_dataset.json --output test_results.json --limit 5

# 2. Evaluate test results
python evaluate_pairwise_metrics.py test_results.json

# 3. If satisfied, run on full dataset
python run_detection_on_dataset.py --dataset agentic_dataset.json --output full_pairwise_results.json

# 4. Generate final evaluation report
python evaluate_pairwise_metrics.py full_pairwise_results.json --output final_evaluation.json
```

## Citation

If you use these metrics in your research, please cite the MAVUL paper:

```bibtex
@misc{li2025mavulmultiagentvulnerabilitydetection,
  title={MAVUL: Multi-Agent Vulnerability Detection via Contextual Reasoning and Interactive Refinement}, 
  author={Youpeng Li and Kartik Joshi and Xinda Wang and Eric Wong},
  year={2025},
  eprint={2510.00317},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2510.00317}
}
```


