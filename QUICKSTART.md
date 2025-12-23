# CoVulPecker - Quick Commands

## Setup (One Time)
```bash
source venv/bin/activate
export JAVA_HOME=/opt/homebrew/opt/openjdk
pip install -r requirements.txt
pip install node2vec networkx gensim torch transformers torch_geometric torch_scatter
```

## Step 1: Build Embeddings from Dataset
```bash
# Test with 5 samples first
python build_dataset_embeddings.py --dataset agentic_dataset.json --limit 5 --device cpu --output_dir embeddings_test

# Full dataset (CPU)
python build_dataset_embeddings.py --dataset agentic_dataset.json --device cpu --output_dir embeddings_output

# Full dataset (GPU - faster)
python build_dataset_embeddings.py --dataset agentic_dataset.json --device cuda:0 --output_dir embeddings_output
```

Output: `embeddings_output/` folder with `.pt` files

## Step 2: Train Model (Optional)
```bash
jupyter notebook classifier.ipynb
# Follow notebook to train and save model to models/best_model_fold5.pt
```

## Step 3: Run Detection Pipeline
```bash
# Setup .env file first
cat > .env << EOF
LLM_API_BASE_URL=https://your-llm-api.com/v1
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_API_KEY=not-needed
EOF

# Run detection on single file
python main.py example.c
python main.py your_file.c

# Run detection on ENTIRE dataset
python run_detection_on_dataset.py --dataset agentic_dataset.json --output detection_results.json

# Test with first 5 samples
python run_detection_on_dataset.py --dataset agentic_dataset.json --limit 5 --output detection_test.json
```

Output: 
- Single file: Console + `logs/` folder
- Dataset: `detection_results.json`

## Alternative: Single File Embeddings (No Dataset)
```bash
python precompute_embeddings.py --input example.c --output_dir embeddings/ --device cpu
```

## Quick Checks
```bash
# Verify setup
./joern-graph/joern/joern-parse --help
python -c "import torch, transformers, node2vec; print('All OK')"

# Check results
ls embeddings_output/
ls models/
ls logs/
```

