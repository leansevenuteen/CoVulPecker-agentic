#!/bin/bash
# Test Phase 1 improvements on 10 samples

echo "=========================================="
echo "Testing Phase 1 Prompt Improvements"
echo "=========================================="
echo ""
echo "Running detection on 10 samples..."
echo ""

# Run on 10 samples
python run_detection_on_dataset.py \
    --dataset agentic_dataset.json \
    --output test_phase1_results.json \
    --limit 10 \
    --max_iterations 3

echo ""
echo "=========================================="
echo "Evaluating Phase 1 Results"
echo "=========================================="
echo ""

# Evaluate results
python evaluate_pairwise_metrics.py test_phase1_results.json --output test_phase1_evaluation.json

echo ""
echo "=========================================="
echo "Phase 1 Testing Complete!"
echo "=========================================="
echo ""
echo "Files created:"
echo "  - test_phase1_results.json (raw results)"
echo "  - test_phase1_evaluation.json (metrics)"
echo "  - test_phase1_evaluation.txt (human-readable)"
echo ""
echo "Compare with baseline:"
echo "  Baseline P-C: 2.4% (6/250)"
echo "  Baseline P-V: 85.6% (214/250)"
echo ""
echo "Target after Phase 1:"
echo "  Target P-C: 15-25% (better distinction)"
echo "  Target P-V: 60-70% (reduced false positives)"
echo ""

