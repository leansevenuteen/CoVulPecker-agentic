#!/bin/bash
# Quick Start - Embedding Pipeline
# Run this after reviewing EMBEDDING_PIPELINE.md

echo "=================================================="
echo "Embedding Pipeline - Quick Start"
echo "Matches attention-covul.ipynb exactly"
echo "=================================================="
echo ""

# Check dependencies
echo "Step 1: Checking dependencies..."
python3 -c "import torch; import transformers; import networkx; import node2vec; print('✓ All dependencies installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip install torch transformers networkx node2vec tqdm
fi
echo ""

# Check dataset
echo "Step 2: Checking dataset..."
if [ ! -f "agentic_dataset.json" ]; then
    echo "❌ agentic_dataset.json not found!"
    echo "   Please ensure your dataset is in the project root."
    exit 1
fi
echo "✓ Dataset found: agentic_dataset.json"
echo ""

# Check Joern
echo "Step 3: Checking Joern installation..."
if [ ! -f "joern-graph/joern/joern-parse" ]; then
    echo "⚠️  Joern not found at joern-graph/joern/"
    echo "   The script will use fallback graph generation."
    echo "   For better results, install Joern from https://joern.io/"
else
    echo "✓ Joern found"
fi
echo ""

# Test run
echo "Step 4: Running test (first 5 samples)..."
echo "Command: python build_dataset_embeddings.py --dataset agentic_dataset.json --limit 5 --device cpu --output_dir embeddings_test"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

python build_dataset_embeddings.py \
    --dataset agentic_dataset.json \
    --limit 5 \
    --device cpu \
    --output_dir embeddings_test

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test successful!"
    echo ""
    echo "Step 5: Verifying output..."
    python test_embeddings.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================================="
        echo "✅ Everything works!"
        echo "=================================================="
        echo ""
        echo "Next: Process full dataset"
        echo ""
        echo "For CPU (slower but works everywhere):"
        echo "  python build_dataset_embeddings.py --dataset agentic_dataset.json --device cpu"
        echo ""
        echo "For GPU (faster, requires CUDA):"
        echo "  python build_dataset_embeddings.py --dataset agentic_dataset.json --device cuda:0"
        echo ""
        echo "Output will be saved to: embeddings_output/"
        echo ""
        echo "See EMBEDDING_PIPELINE.md for detailed documentation."
        echo ""
    else
        echo "⚠️  Verification failed. Check the output above."
    fi
else
    echo "❌ Test failed. Check the error messages above."
    exit 1
fi
