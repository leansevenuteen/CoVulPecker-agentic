"""
Compare Phase 1 improvements against baseline results.

Usage:
    python compare_improvements.py baseline_results.json test_phase1_results.json
"""
import json
import sys
from collections import defaultdict


def load_results(file_path):
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_metrics(results):
    """Calculate pairwise metrics from results."""
    successful = [r for r in results if r.get('status') == 'success']
    
    if not successful:
        return None
    
    categories = defaultdict(int)
    for result in successful:
        category = result.get('pairwise_category', 'UNKNOWN')
        categories[category] += 1
    
    total = len(successful)
    
    return {
        'total': total,
        'P-C': {'count': categories['P-C'], 'pct': categories['P-C']/total*100 if total > 0 else 0},
        'P-V': {'count': categories['P-V'], 'pct': categories['P-V']/total*100 if total > 0 else 0},
        'P-B': {'count': categories['P-B'], 'pct': categories['P-B']/total*100 if total > 0 else 0},
        'P-R': {'count': categories['P-R'], 'pct': categories['P-R']/total*100 if total > 0 else 0},
    }


def print_comparison(baseline_metrics, phase1_metrics):
    """Print comparison between baseline and Phase 1."""
    print("=" * 80)
    print("PHASE 1 IMPROVEMENTS - COMPARISON REPORT")
    print("=" * 80)
    print()
    
    print(f"Baseline: {baseline_metrics['total']} samples")
    print(f"Phase 1:  {phase1_metrics['total']} samples")
    print()
    
    print("-" * 80)
    print(f"{'Metric':<20} {'Baseline':<20} {'Phase 1':<20} {'Change':<20}")
    print("-" * 80)
    
    for metric in ['P-C', 'P-V', 'P-B', 'P-R']:
        baseline_val = baseline_metrics[metric]['pct']
        phase1_val = phase1_metrics[metric]['pct']
        change = phase1_val - baseline_val
        
        baseline_str = f"{baseline_val:5.1f}% ({baseline_metrics[metric]['count']:3d})"
        phase1_str = f"{phase1_val:5.1f}% ({phase1_metrics[metric]['count']:3d})"
        
        # Determine if change is good or bad
        if metric == 'P-C':
            # Higher is better
            indicator = "✓" if change > 0 else "✗" if change < 0 else "="
            change_str = f"{change:+5.1f}% {indicator}"
        else:
            # Lower is better for P-V, P-B, P-R
            indicator = "✓" if change < 0 else "✗" if change > 0 else "="
            change_str = f"{change:+5.1f}% {indicator}"
        
        print(f"{metric:<20} {baseline_str:<20} {phase1_str:<20} {change_str:<20}")
    
    print("-" * 80)
    print()
    
    # Calculate improvement summary
    p_c_improvement = phase1_metrics['P-C']['pct'] - baseline_metrics['P-C']['pct']
    p_v_reduction = baseline_metrics['P-V']['pct'] - phase1_metrics['P-V']['pct']
    
    print("SUMMARY:")
    print(f"  P-C Improvement:      {p_c_improvement:+.1f} percentage points")
    print(f"  P-V Reduction:        {p_v_reduction:+.1f} percentage points")
    print()
    
    # Assessment
    print("ASSESSMENT:")
    if p_c_improvement > 10:
        print("  ✓ Excellent P-C improvement (>10%)")
    elif p_c_improvement > 5:
        print("  ✓ Good P-C improvement (5-10%)")
    elif p_c_improvement > 0:
        print("  ~ Modest P-C improvement (<5%)")
    else:
        print("  ✗ P-C decreased or no change")
    
    if p_v_reduction > 20:
        print("  ✓ Excellent P-V reduction (>20%)")
    elif p_v_reduction > 10:
        print("  ✓ Good P-V reduction (10-20%)")
    elif p_v_reduction > 0:
        print("  ~ Modest P-V reduction (<10%)")
    else:
        print("  ✗ P-V increased or no change")
    
    print()
    print("=" * 80)
    
    # Detailed analysis
    print()
    print("DETAILED ANALYSIS:")
    print()
    
    # Check if false positives reduced
    if p_v_reduction > 0:
        print(f"✓ Phase 1 successfully reduced false positives on post-patch code")
        print(f"  Before: {baseline_metrics['P-V']['count']} samples flagged as vulnerable after patching")
        print(f"  After:  {phase1_metrics['P-V']['count']} samples flagged as vulnerable after patching")
        print(f"  Improvement: {baseline_metrics['P-V']['count'] - phase1_metrics['P-V']['count']} fewer false positives")
    else:
        print(f"⚠ Phase 1 did not reduce false positives (may need more samples or further tuning)")
    
    print()
    
    # Check correct classifications
    if p_c_improvement > 0:
        print(f"✓ Phase 1 improved correct pairwise classifications")
        print(f"  Before: {baseline_metrics['P-C']['count']} correctly classified pairs")
        print(f"  After:  {phase1_metrics['P-C']['count']} correctly classified pairs")
        print(f"  Improvement: {phase1_metrics['P-C']['count'] - baseline_metrics['P-C']['count']} more correct")
    else:
        print(f"⚠ P-C did not improve (this could indicate prompts need further refinement)")
    
    print()
    print("=" * 80)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_improvements.py baseline_results.json phase1_results.json")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    phase1_path = sys.argv[2]
    
    print(f"Loading baseline from: {baseline_path}")
    baseline_results = load_results(baseline_path)
    baseline_metrics = calculate_metrics(baseline_results)
    
    print(f"Loading Phase 1 from: {phase1_path}")
    phase1_results = load_results(phase1_path)
    phase1_metrics = calculate_metrics(phase1_results)
    
    if not baseline_metrics or not phase1_metrics:
        print("Error: No successful results to compare")
        sys.exit(1)
    
    print()
    print_comparison(baseline_metrics, phase1_metrics)
    
    # Save comparison to file
    comparison = {
        'baseline': baseline_metrics,
        'phase1': phase1_metrics,
        'improvements': {
            'p_c_change': phase1_metrics['P-C']['pct'] - baseline_metrics['P-C']['pct'],
            'p_v_change': phase1_metrics['P-V']['pct'] - baseline_metrics['P-V']['pct'],
            'p_b_change': phase1_metrics['P-B']['pct'] - baseline_metrics['P-B']['pct'],
            'p_r_change': phase1_metrics['P-R']['pct'] - baseline_metrics['P-R']['pct'],
        }
    }
    
    output_file = 'phase1_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print()
    print(f"Comparison saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()

