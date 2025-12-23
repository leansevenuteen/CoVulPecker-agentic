"""
Run vulnerability detection pipeline on entire dataset.

This script processes all samples in agentic_dataset.json through
the 5-stage detection pipeline and saves results.
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.graph import vulnerability_detector
from src.config import config
from src.logger import logger


def run_detection_on_dataset(
    dataset_path: str = "agentic_dataset.json",
    output_path: str = "detection_results.json",
    max_iterations: int = 3,
    limit: int = None
):
    """
    Run detection pipeline on all dataset samples.
    
    Args:
        dataset_path: Path to dataset JSON
        output_path: Path to save results
        max_iterations: Max critic iterations per sample
        limit: Process only first N samples (None = all)
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    if limit:
        dataset = dataset[:limit]
        print(f"Limited to first {limit} samples")
    
    print(f"Total samples: {len(dataset)}")
    print(f"LLM API: {config.LLM_API_BASE_URL}")
    print("-" * 70)
    
    results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        sample_id = sample.get('id', idx)
        pre_patch_code = sample.get('pre_patch', sample.get('code', ''))
        post_patch_code = sample.get('post_patch', '')
        true_label = sample.get('label', None)
        cwe_id = sample.get('cwe_id', 'Unknown')
        
        print(f"\n[{idx+1}/{len(dataset)}] Sample {sample_id} (CWE: {cwe_id})")
        
        try:
            # Run detection pipeline on PRE-PATCH (vulnerable) code
            print(f"  Analyzing PRE-PATCH code...")
            pre_state = {
                "source_code": pre_patch_code,
                "classification": None,
                "detection": None,
                "reasoning": None,
                "critic": None,
                "critic_feedback_history": [],
                "verification": None,
                "iteration_count": 1,
                "max_iterations": max_iterations,
                "current_stage": "start",
                "is_complete": False,
            }
            
            pre_final_state = vulnerability_detector.invoke(pre_state)
            
            pre_prediction = {
                "predicted_label": pre_final_state.get("classification").label if pre_final_state.get("classification") else None,
                "confidence": pre_final_state.get("classification").confidence if pre_final_state.get("classification") else None,
                "vulnerabilities_found": len(pre_final_state.get("detection").vulnerabilities) if pre_final_state.get("detection") else 0,
                "vulnerability_confirmed": pre_final_state.get("verification").vulnerability_confirmed if pre_final_state.get("verification") else False,
                "critic_decision": pre_final_state.get("critic").decision if pre_final_state.get("critic") else None,
            }
            
            print(f"    PRE-PATCH: {pre_prediction['predicted_label']} ({pre_prediction['confidence']:.2%})")
            
            # Run detection pipeline on POST-PATCH (fixed) code
            print(f"  Analyzing POST-PATCH code...")
            post_state = {
                "source_code": post_patch_code,
                "classification": None,
                "detection": None,
                "reasoning": None,
                "critic": None,
                "critic_feedback_history": [],
                "verification": None,
                "iteration_count": 1,
                "max_iterations": max_iterations,
                "current_stage": "start",
                "is_complete": False,
            }
            
            post_final_state = vulnerability_detector.invoke(post_state)
            
            post_prediction = {
                "predicted_label": post_final_state.get("classification").label if post_final_state.get("classification") else None,
                "confidence": post_final_state.get("classification").confidence if post_final_state.get("classification") else None,
                "vulnerabilities_found": len(post_final_state.get("detection").vulnerabilities) if post_final_state.get("detection") else 0,
                "vulnerability_confirmed": post_final_state.get("verification").vulnerability_confirmed if post_final_state.get("verification") else False,
                "critic_decision": post_final_state.get("critic").decision if post_final_state.get("critic") else None,
            }
            
            print(f"    POST-PATCH: {post_prediction['predicted_label']} ({post_prediction['confidence']:.2%})")
            
            # Determine pairwise category
            pre_label = pre_prediction['predicted_label']
            post_label = post_prediction['predicted_label']
            
            if pre_label == "vulnerable" and post_label == "non-vulnerable":
                pairwise_category = "P-C"  # Perfect-Correct
            elif pre_label == "vulnerable" and post_label == "vulnerable":
                pairwise_category = "P-V"  # Perfect-Vulnerable (FP on post)
            elif pre_label == "non-vulnerable" and post_label == "non-vulnerable":
                pairwise_category = "P-B"  # Perfect-Benign (FN on pre)
            elif pre_label == "non-vulnerable" and post_label == "vulnerable":
                pairwise_category = "P-R"  # Perfect-Reverse (worst case)
            else:
                pairwise_category = "UNKNOWN"
            
            print(f"  Pairwise Category: {pairwise_category}")
            
            # Build result with pairwise structure
            result = {
                "sample_id": sample_id,
                "cwe_id": cwe_id,
                "true_label": true_label,
                "pre_patch_prediction": pre_prediction,
                "post_patch_prediction": post_prediction,
                "pairwise_category": pairwise_category,
                "iterations": pre_final_state.get("iteration_count", 1),
                "status": "success"
            }
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            result = {
                "sample_id": sample_id,
                "cwe_id": cwe_id,
                "true_label": true_label,
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
        
        # Save intermediate results every 10 samples
        if (idx + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved intermediate results to {output_path}")
    
    # Save final results
    print("\n" + "=" * 70)
    print("Saving final results...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    # Pairwise metrics
    p_c = [r for r in successful if r.get('pairwise_category') == 'P-C']
    p_v = [r for r in successful if r.get('pairwise_category') == 'P-V']
    p_b = [r for r in successful if r.get('pairwise_category') == 'P-B']
    p_r = [r for r in successful if r.get('pairwise_category') == 'P-R']
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total samples:           {len(results)}")
    print(f"Successful:              {len(successful)}")
    print(f"Failed:                  {len(failed)}")
    print(f"\nPAIRWISE METRICS (MAVUL)")
    print(f"{'=' * 70}")
    if successful:
        print(f"P-C (Perfect-Correct):       {len(p_c):4d} ({len(p_c)/len(successful)*100:5.1f}%) <- Higher is better")
        print(f"P-V (Perfect-Vulnerable):    {len(p_v):4d} ({len(p_v)/len(successful)*100:5.1f}%) <- Lower is better")
        print(f"P-B (Perfect-Benign):        {len(p_b):4d} ({len(p_b)/len(successful)*100:5.1f}%) <- Lower is better")
        print(f"P-R (Perfect-Reverse):       {len(p_r):4d} ({len(p_r)/len(successful)*100:5.1f}%) <- Lower is better")
        print(f"\nPairwise Accuracy (P-C):     {len(p_c)/len(successful)*100:.2f}%")
    print(f"\nResults saved to: {output_path}")
    print(f"{'=' * 70}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run detection on entire dataset")
    parser.add_argument('--dataset', type=str, default='agentic_dataset.json', 
                       help='Path to dataset JSON')
    parser.add_argument('--output', type=str, default='detection_results.json',
                       help='Output file for results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Process only first N samples (for testing)')
    parser.add_argument('--max_iterations', type=int, default=3,
                       help='Max critic iterations per sample')
    
    args = parser.parse_args()
    
    try:
        results = run_detection_on_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            max_iterations=args.max_iterations,
            limit=args.limit
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may be saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

