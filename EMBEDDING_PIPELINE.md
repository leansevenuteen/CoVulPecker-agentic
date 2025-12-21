# Embedding Pipeline - Matching attention-covul.ipynb

This document explains the embedding pipeline that matches the Kaggle notebook exactly.

## Critical Differences from Previous Implementation

### 1. **CodeT5 Node Tokenization** (MOST IMPORTANT!)

**Notebook Approach (CORRECT):**
```python
# Extract node labels from graph
code_sequence = [str(node['label']) for node in nodes.values()]

# Tokenize EACH NODE LABEL separately  
inputs = tokenizer(code_sequence, padding=True, truncation=True, max_length=128)

# Result: [num_nodes, 768] embeddings, one per node
```

**Previous Implementation (WRONG):**
```python
# Tokenize entire source code once
inputs = tokenizer(source_code, max_length=512, truncation=True)

# Replicate for all nodes
embeddings = np.tile(embeddings, (num_nodes, 1))  

# Result: Same embedding copied for all nodes (loses granularity!)
```

**Impact:** The notebook's per-node tokenization captures the semantic meaning of each individual graph node, while the previous approach just copies one global embedding.

### 2. **Embedding Extraction Method**

**Notebook:**
```python
pooled_output = outputs.last_hidden_state[:, 0]  # First token
```

**Previous:**
```python
embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
```

**Impact:** First token focuses on [CLS]-like representation, mean pooling averages all tokens.

### 3. **Node2Vec Parameters**

**Notebook (Fixed):**
- `p=1, q=2` - Fixed parameters favoring exploration
- `walk_length=100` - Long walks
- `num_walks=10` - Moderate number of walks
- `max_length=128` - For tokenization

**Previous (Configurable):**
- Parameters passed as arguments
- `max_length=512` default
- `epochs=5, batch_words=4` in fit()

### 4. **Input/Output Format**

**Notebook Input:**
- Pre-processed pickle file with graph data
- Columns: `nodes`, `adj_matrix`, `label`, `split`, `cwe_id`

**Notebook Output:**
```python
# Lists of tensors for ENTIRE dataset:
torch.save(adj_matrices, 'adj_matrices.pt')           # List of [2, E] tensors
torch.save(n2v_embeddings, 'n2v_node_embeddings.pt')  # List of [N, 200] tensors
torch.save(codet5_embeddings, 'aggregated_codet5_embeddings.pt')  # List of [N, 768] tensors

# Metadata JSON
json.dump({'labels': labels, 'partition': splits}, 'metadata.json')
```

**Previous Output (Per-File):**
```python
# 3 files per source file:
torch.save(graph_embeddings, f'{name}_graph_embeddings.pt')  # [N, 200]
torch.save(token_embeddings, f'{name}_token_embeddings.pt')  # [N, 768]  
torch.save(edge_index, f'{name}_edge_index.pt')              # [2, E]
```

## Pipeline Flow

```
┌──────────────┐
│ Source Code  │
│   (string)   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│   Joern CPG Extract      │
│   - Parse to .bin        │
│   - Export to .dot       │
│   - Extract nodes/edges  │
└──────┬───────────────────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌─────────────────┐    ┌──────────────────────┐
│  Node Labels    │    │  Graph Structure     │
│  ['func', 'var']│    │  edge_matrix [2, E]  │
└─────┬───────────┘    └─────┬────────────────┘
      │                      │
      │                      ▼
      │              ┌────────────────────┐
      │              │   Node2Vec         │
      │              │   p=1, q=2         │
      │              │   walk_len=100     │
      │              └────┬───────────────┘
      │                   │
      ▼                   ▼
┌─────────────────┐  ┌──────────────────┐
│  CodeT5 Encode  │  │  Graph Embeddings│
│  Each Node      │  │  [N, 200]        │
│  max_len=128    │  └──────────────────┘
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│ Token Embeddings│
│  [:, 0] (first) │
│  [N, 768]       │
└─────────────────┘
```

## Usage

### Option 1: Build Complete Dataset (Recommended)

This matches the notebook exactly - processes all samples and saves as lists:

```bash
# Install dependencies
pip install torch transformers node2vec networkx

# Process dataset
python build_dataset_embeddings.py \
    --dataset agentic_dataset.json \
    --output_dir embeddings_output \
    --device cuda:0 \
    --limit 10  # Optional: test with first 10 samples
```

**Output:**
```
embeddings_output/
  ├── adj_matrices.pt                      # List of edge indices
  ├── n2v_node_embeddings.pt              # List of Node2Vec embeddings
  ├── aggregated_codet5_embeddings.pt     # List of CodeT5 embeddings
  └── dataset_metadata.json               # Labels and splits
```

### Option 2: Use Precomputed Embeddings

For single-file processing (testing/development):

```bash
python precompute_embeddings.py \
    --input example.c \
    --output_dir embeddings/ \
    --device cpu
```

**Note:** This uses the OLD implementation and should be updated to match notebook logic.

## Loading Embeddings for Training

```python
import torch
import json

# Load embeddings
adj_matrices = torch.load('embeddings_output/adj_matrices.pt')
n2v_embeddings = torch.load('embeddings_output/n2v_node_embeddings.pt')
codet5_embeddings = torch.load('embeddings_output/aggregated_codet5_embeddings.pt')

# Load metadata
with open('embeddings_output/dataset_metadata.json') as f:
    metadata = json.load(f)
    labels = metadata['labels']
    splits = metadata['partition']

# Access individual samples
for i in range(len(labels)):
    adj = adj_matrices[i]        # [2, num_edges]
    n2v = n2v_embeddings[i]      # [num_nodes, 200]
    codet5 = codet5_embeddings[i] # [num_nodes, 768]
    label = labels[i]             # 0 or 1
    split = splits[i]             # 'train', 'valid', 'test'
    
    # Your training code here...
```

## Verification

Check your embeddings match the expected format:

```python
import torch

# Load
adj = torch.load('embeddings_output/adj_matrices.pt')
n2v = torch.load('embeddings_output/n2v_node_embeddings.pt')
codet5 = torch.load('embeddings_output/aggregated_codet5_embeddings.pt')

print(f"Number of samples: {len(adj)}")
print(f"First sample:")
print(f"  - Adjacency: {adj[0].shape}")      # Should be [2, num_edges]
print(f"  - Node2Vec: {n2v[0].shape}")       # Should be [num_nodes, 200]
print(f"  - CodeT5: {codet5[0].shape}")      # Should be [num_nodes, 768]
print(f"  - Nodes match: {n2v[0].shape[0] == codet5[0].shape[0]}")
```

Expected output:
```
Number of samples: 250
First sample:
  - Adjacency: torch.Size([2, 140])
  - Node2Vec: torch.Size([129, 200])
  - CodeT5: torch.Size([129, 768])
  - Nodes match: True
```

## Model Loading

The notebook uses a pre-trained CodeT5 classifier model:

```python
CODELM_PATH = '/kaggle/input/primevul-codet5-ft/pytorch/default/1/codet5_classifier.pth'

# Load model weights
codet5_model = CodeT5Classifier(model_name="Salesforce/codet5-base", n_classes=2)
codet5_model.load_state_dict(torch.load(CODELM_PATH))
```

Your model weights should be saved at:
```
models/best_model_fold5.pt
```

## Troubleshooting

### Issue: "Joern not found"
```bash
# Install Joern (macOS)
brew install joernio/tap/joern

# Or download from https://joern.io/
# Add to PATH in your script
```

### Issue: "CUDA out of memory"
```bash
# Use CPU instead
python build_dataset_embeddings.py --device cpu

# Or process in smaller batches
python build_dataset_embeddings.py --limit 50
```

### Issue: "Node labels missing"
The Joern extraction might fail. Check:
1. Source code is valid C/C++
2. Joern is properly installed
3. Temporary directory has write permissions

### Issue: "Embeddings don't match"
Verify you're using:
- CodeT5: `Salesforce/codet5-base`
- Node2Vec: `p=1, q=2, walk_length=100, num_walks=10`
- Max length: `128` (not 512)
- First token: `[:, 0]` (not mean pooling)

## Summary

**Key Takeaways:**

1. **Per-node tokenization** is critical - tokenize node labels, not source code
2. **First token** embedding, not mean pooling
3. **Fixed Node2Vec params**: p=1, q=2, walk_length=100
4. **Batch output format**: Lists of tensors for entire dataset
5. Use `build_dataset_embeddings.py` for production (matches notebook exactly)

This pipeline ensures your embeddings match the attention-covul.ipynb notebook exactly, which is critical for reproducing the model's performance.

