"""
Evaluate MAVUL Pairwise Metrics from Detection Results.

This script loads detection results that contain both pre-patch and post-patch
predictions and calculates the MAVUL pairwise metrics (P-C, P-V, P-B, P-R).
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def calculate_pairwise_category(pre_label: str, post_label: str, pre_true: int = 1, post_true: int = 0) -> str:
    """
    Determine pairwise category by comparing predictions to true labels.
    
    Args:
        pre_label: Predicted label for pre-patch code ("vulnerable" or "clean")
        post_label: Predicted label for post-patch code ("vulnerable" or "clean")
        pre_true: True label for pre-patch (1=vulnerable, 0=clean, default=1)
        post_true: True label for post-patch (1=vulnerable, 0=clean, default=0)
        
    Returns:
        Pairwise category: P-C, P-V, P-B, P-R, or UNKNOWN
        
    Categories based on MAVUL metrics:
        P-C: TP (pre) + TN (post) - Both predictions correct
        P-V: TP (pre) + FP (post) - Pre correct, post wrong
        P-B: FN (pre) + TN (post) - Pre wrong, post correct
        P-R: FN (pre) + FP (post) - Both predictions wrong
    """
    # Convert labels to boolean for easier comparison
    pre_pred = (pre_label == "vulnerable")
    post_pred = (post_label == "vulnerable")
    pre_true_vuln = (pre_true == 1)
    post_true_vuln = (post_true == 1)  # Should always be 0 for postpatch
    
    # Determine TP/TN/FP/FN for prepatch
    pre_tp = pre_pred and pre_true_vuln      # True Positive: predicted vulnerable, is vulnerable
    pre_fn = not pre_pred and pre_true_vuln  # False Negative: predicted clean, is vulnerable
    
    # Determine TP/TN/FP/FN for postpatch
    post_tn = not post_pred and not post_true_vuln  # True Negative: predicted clean, is clean
    post_fp = post_pred and not post_true_vuln      # False Positive: predicted vulnerable, is clean
    
    # Categorize based on both results
    if pre_tp and post_tn:
        return "P-C"  # Perfect-Correct: both correct
    elif pre_tp and post_fp:
        return "P-V"  # Perfect-Vulnerable: pre correct, post FP
    elif pre_fn and post_tn:
        return "P-B"  # Perfect-Benign: pre FN, post correct
    elif pre_fn and post_fp:
        return "P-R"  # Perfect-Reverse: both wrong
    else:
        return "UNKNOWN"


def evaluate_pairwise_metrics(
    results_path: str,
    output_path: str = None,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Evaluate pairwise metrics from detection results.
    
    Args:
        results_path: Path to detection results JSON
        output_path: Optional path to save evaluation report
        detailed: Whether to include detailed per-sample breakdown
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Filter successful results
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    if not successful:
        print("No successful results to evaluate!")
        return {}
    
    print(f"Total results: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print("-" * 70)
    
    # Calculate or extract pairwise categories
    categorized_results = []
    for result in successful:
        # Extract predictions
        pre_label = result.get('pre_patch_prediction', {}).get('predicted_label')
        post_label = result.get('post_patch_prediction', {}).get('predicted_label')
        
        # Extract or infer true labels
        # Priority: explicit pre/post labels > single label field > default (1, 0)
        if 'pre_patch_true_label' in result and 'post_patch_true_label' in result:
            pre_true = result['pre_patch_true_label']
            post_true = result['post_patch_true_label']
        elif 'true_label' in result:
            # Old format: assume single label is for prepatch, postpatch is always 0
            pre_true = result['true_label']
            post_true = 0
        else:
            # Default: prepatch is vulnerable (1), postpatch is clean (0)
            pre_true = 1
            post_true = 0
        
        # Calculate category using true labels
        if pre_label and post_label:
            category = calculate_pairwise_category(pre_label, post_label, pre_true, post_true)
        else:
            category = "UNKNOWN"
        
        categorized_results.append({
            **result,
            'pairwise_category': category,
            'pre_patch_true_label': pre_true,
            'post_patch_true_label': post_true
        })
    
    # Count categories
    category_counts = defaultdict(int)
    for result in categorized_results:
        category = result['pairwise_category']
        category_counts[category] += 1
    
    # Calculate metrics
    total = len(categorized_results)
    p_c_count = category_counts['P-C']
    p_v_count = category_counts['P-V']
    p_b_count = category_counts['P-B']
    p_r_count = category_counts['P-R']
    unknown_count = category_counts['UNKNOWN']
    
    # Pairwise accuracy is the P-C percentage
    pairwise_accuracy = (p_c_count / total * 100) if total > 0 else 0
    
    # Calculate per-CWE breakdown
    cwe_breakdown = defaultdict(lambda: defaultdict(int))
    for result in categorized_results:
        cwe = result.get('cwe_id', 'Unknown')
        category = result['pairwise_category']
        cwe_breakdown[cwe][category] += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("MAVUL PAIRWISE METRICS EVALUATION")
    print("=" * 70)
    print(f"Total Pairs Analyzed: {total}")
    print()
    print(f"P-C (Perfect-Correct):       {p_c_count:4d} ({p_c_count/total*100:5.1f}%)  <- Higher is better")
    print(f"P-V (Perfect-Vulnerable):    {p_v_count:4d} ({p_v_count/total*100:5.1f}%)  <- Lower is better")
    print(f"P-B (Perfect-Benign):        {p_b_count:4d} ({p_b_count/total*100:5.1f}%)  <- Lower is better")
    print(f"P-R (Perfect-Reverse):       {p_r_count:4d} ({p_r_count/total*100:5.1f}%)  <- Lower is better")
    if unknown_count > 0:
        print(f"UNKNOWN:                     {unknown_count:4d} ({unknown_count/total*100:5.1f}%)")
    print()
    print(f"Pairwise Accuracy (P-C): {pairwise_accuracy:.2f}%")
    print("=" * 70)
    
    # Print per-CWE breakdown
    if len(cwe_breakdown) > 1 and detailed:
        print("\nPER-CWE BREAKDOWN")
        print("=" * 70)
        for cwe in sorted(cwe_breakdown.keys()):
            counts = cwe_breakdown[cwe]
            cwe_total = sum(counts.values())
            cwe_p_c = counts['P-C']
            cwe_accuracy = (cwe_p_c / cwe_total * 100) if cwe_total > 0 else 0
            
            print(f"\n{cwe} (n={cwe_total}):")
            print(f"  P-C: {counts['P-C']:3d} ({counts['P-C']/cwe_total*100:5.1f}%)")
            print(f"  P-V: {counts['P-V']:3d} ({counts['P-V']/cwe_total*100:5.1f}%)")
            print(f"  P-B: {counts['P-B']:3d} ({counts['P-B']/cwe_total*100:5.1f}%)")
            print(f"  P-R: {counts['P-R']:3d} ({counts['P-R']/cwe_total*100:5.1f}%)")
            print(f"  Accuracy: {cwe_accuracy:.2f}%")
    
    # Build evaluation report
    evaluation_report = {
        "total_samples": total,
        "metrics": {
            "P-C": {
                "count": p_c_count,
                "percentage": round(p_c_count/total*100, 2),
                "description": "Perfect-Correct (both predictions correct)"
            },
            "P-V": {
                "count": p_v_count,
                "percentage": round(p_v_count/total*100, 2),
                "description": "Perfect-Vulnerable (missed that patch fixed vulnerability)"
            },
            "P-B": {
                "count": p_b_count,
                "percentage": round(p_b_count/total*100, 2),
                "description": "Perfect-Benign (missed vulnerability entirely)"
            },
            "P-R": {
                "count": p_r_count,
                "percentage": round(p_r_count/total*100, 2),
                "description": "Perfect-Reverse (completely backwards)"
            }
        },
        "pairwise_accuracy": round(pairwise_accuracy, 2),
        "cwe_breakdown": {
            cwe: {
                "total": sum(counts.values()),
                "P-C": counts['P-C'],
                "P-V": counts['P-V'],
                "P-B": counts['P-B'],
                "P-R": counts['P-R'],
                "accuracy": round((counts['P-C'] / sum(counts.values()) * 100), 2)
            }
            for cwe, counts in cwe_breakdown.items()
        }
    }
    
    if detailed:
        evaluation_report["detailed_results"] = categorized_results
    
    # Save evaluation report
    if output_path:
        print(f"\nSaving evaluation report to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        print(f"Report saved!")
        
        # Also save a human-readable text version
        txt_path = Path(output_path).with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("MAVUL PAIRWISE METRICS EVALUATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Pairs Analyzed: {total}\n\n")
            f.write(f"P-C (Perfect-Correct):       {p_c_count:4d} ({p_c_count/total*100:5.1f}%)  <- Higher is better\n")
            f.write(f"P-V (Perfect-Vulnerable):    {p_v_count:4d} ({p_v_count/total*100:5.1f}%)  <- Lower is better\n")
            f.write(f"P-B (Perfect-Benign):        {p_b_count:4d} ({p_b_count/total*100:5.1f}%)  <- Lower is better\n")
            f.write(f"P-R (Perfect-Reverse):       {p_r_count:4d} ({p_r_count/total*100:5.1f}%)  <- Lower is better\n")
            f.write(f"\nPairwise Accuracy (P-C): {pairwise_accuracy:.2f}%\n")
            f.write("=" * 70 + "\n\n")
            
            if len(cwe_breakdown) > 1:
                f.write("\nPER-CWE BREAKDOWN\n")
                f.write("=" * 70 + "\n")
                for cwe in sorted(cwe_breakdown.keys()):
                    counts = cwe_breakdown[cwe]
                    cwe_total = sum(counts.values())
                    cwe_accuracy = (counts['P-C'] / cwe_total * 100) if cwe_total > 0 else 0
                    
                    f.write(f"\n{cwe} (n={cwe_total}):\n")
                    f.write(f"  P-C: {counts['P-C']:3d} ({counts['P-C']/cwe_total*100:5.1f}%)\n")
                    f.write(f"  P-V: {counts['P-V']:3d} ({counts['P-V']/cwe_total*100:5.1f}%)\n")
                    f.write(f"  P-B: {counts['P-B']:3d} ({counts['P-B']/cwe_total*100:5.1f}%)\n")
                    f.write(f"  P-R: {counts['P-R']:3d} ({counts['P-R']/cwe_total*100:5.1f}%)\n")
                    f.write(f"  Accuracy: {cwe_accuracy:.2f}%\n")
        
        print(f"Text report saved to {txt_path}")
    
    return evaluation_report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate MAVUL pairwise metrics from detection results"
    )
    parser.add_argument(
        'results',
        type=str,
        help='Path to detection results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pairwise_evaluation.json',
        help='Output path for evaluation report (default: pairwise_evaluation.json)'
    )
    parser.add_argument(
        '--no-detailed',
        action='store_true',
        help='Exclude detailed per-sample breakdown from output'
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_pairwise_metrics(
            results_path=args.results,
            output_path=args.output,
            detailed=not args.no_detailed
        )
    except FileNotFoundError:
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in results file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




