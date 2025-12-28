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
    
    print("Testing LLM connection...")
    try:
        from src.llm import get_llm
        test_llm = get_llm(temperature=0.1)
        test_response = test_llm.invoke([{"role": "user", "content": "Reply with OK"}])
        print(f"✓ LLM connected: {test_response.content[:50]}")
    except Exception as e:
        print(f"✗ LLM connection failed: {e}")
        print("Pipeline will use fallback values (no LLM calls)")

    results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        sample_id = sample.get('id', idx)
        pre_patch_code = sample.get('pre_patch', sample.get('code', ''))
        post_patch_code = sample.get('post_patch', '')
        cwe_id = sample.get('cwe_id', 'Unknown')
        
        # Extract true labels - handle both old and new dataset formats
        if 'pre_patch_label' in sample and 'post_patch_label' in sample:
            # New format: explicit labels for both
            pre_true_label = sample['pre_patch_label']
            post_true_label = sample['post_patch_label']
        elif 'label' in sample:
            # Old format: single label (assume prepatch=label, postpatch=0)
            pre_true_label = sample['label']
            post_true_label = 0
        else:
            # Default: prepatch is vulnerable, postpatch is clean
            pre_true_label = 1
            post_true_label = 0
        
        print(f"\n[{idx+1}/{len(dataset)}] Sample {sample_id} (CWE: {cwe_id})")
        print(f"  True labels: pre={pre_true_label}, post={post_true_label}")
        
        try:
            # Run detection pipeline on PRE-PATCH (vulnerable) code
            print(f"  Analyzing PRE-PATCH code...")
            pre_state = {
                "source_code": pre_patch_code,
                "code_version": "pre-patch",
                "analysis_context": "You are analyzing the ORIGINAL vulnerable code before any security fixes were applied. Your goal is to identify security vulnerabilities that need to be patched.",
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
            
            # Determine final label from Verification agent (last stage), not Classifier (first stage)
            verification_confirmed = pre_final_state.get("verification").vulnerability_confirmed if pre_final_state.get("verification") else False
            final_label = "vulnerable" if verification_confirmed else "clean"
            
            # Get classifier's initial prediction for reference
            classifier_label = pre_final_state.get("classification").label if pre_final_state.get("classification") else None
            classifier_confidence = pre_final_state.get("classification").confidence if pre_final_state.get("classification") else None
            
            pre_prediction = {
                "predicted_label": final_label,  # Final decision from multi-agent system
                "classifier_initial_label": classifier_label,  # DL classifier's initial prediction
                "classifier_confidence": classifier_confidence,  # DL classifier's confidence
                "vulnerabilities_found": len(pre_final_state.get("detection").vulnerabilities) if pre_final_state.get("detection") else 0,
                "vulnerability_confirmed": verification_confirmed,
                "critic_decision": pre_final_state.get("critic").decision if pre_final_state.get("critic") else None,
            }
            
            print(f"    PRE-PATCH: {pre_prediction['predicted_label']} (DL initial: {classifier_label} {classifier_confidence:.2%}, verified: {verification_confirmed})")
            
            # Run detection pipeline on POST-PATCH (fixed) code
            print(f"  Analyzing POST-PATCH code...")
            post_state = {
                "source_code": post_patch_code,
                "code_version": "post-patch",
                "analysis_context": "You are analyzing the PATCHED/FIXED version of the code. Your goal is to verify that security vulnerabilities have been properly addressed and fixed.",
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
            
            # Determine final label from Verification agent (last stage), not Classifier (first stage)
            verification_confirmed = post_final_state.get("verification").vulnerability_confirmed if post_final_state.get("verification") else False
            final_label = "vulnerable" if verification_confirmed else "clean"
            
            # Get classifier's initial prediction for reference
            classifier_label = post_final_state.get("classification").label if post_final_state.get("classification") else None
            classifier_confidence = post_final_state.get("classification").confidence if post_final_state.get("classification") else None
            
            post_prediction = {
                "predicted_label": final_label,  # Final decision from multi-agent system
                "classifier_initial_label": classifier_label,  # DL classifier's initial prediction
                "classifier_confidence": classifier_confidence,  # DL classifier's confidence
                "vulnerabilities_found": len(post_final_state.get("detection").vulnerabilities) if post_final_state.get("detection") else 0,
                "vulnerability_confirmed": verification_confirmed,
                "critic_decision": post_final_state.get("critic").decision if post_final_state.get("critic") else None,
            }
            
            print(f"    POST-PATCH: {post_prediction['predicted_label']} (DL initial: {classifier_label} {classifier_confidence:.2%}, verified: {verification_confirmed})")
            
            # Determine pairwise category
            pre_label = pre_prediction['predicted_label']
            post_label = post_prediction['predicted_label']
            
            if pre_label == "vulnerable" and post_label == "clean":
                pairwise_category = "P-C"  # Pairwise-Correct
            elif pre_label == "vulnerable" and post_label == "vulnerable":
                pairwise_category = "P-V"  # Pairwise-Vulnerable (FP on post)
            elif pre_label == "clean" and post_label == "clean":
                pairwise_category = "P-B"  # Pairwise-Benign (FN on pre)
            elif pre_label == "clean" and post_label == "vulnerable":
                pairwise_category = "P-R"  # Pairwise-Reverse (worst case)
            else:
                pairwise_category = "UNKNOWN"
            
            print(f"  Pairwise Category: {pairwise_category}")
            
            # Build result with pairwise structure
            result = {
                "sample_id": sample_id,
                "cwe_id": cwe_id,
                "pre_patch_true_label": pre_true_label,
                "post_patch_true_label": post_true_label,
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
                "pre_patch_true_label": pre_true_label,
                "post_patch_true_label": post_true_label,
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

