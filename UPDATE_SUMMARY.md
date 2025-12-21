# Embedding Pipeline Update Summary

## What Was Changed

Updated the embedding pipeline to **exactly match** the attention-covul.ipynb notebook.

## Key Files Created/Modified

### 1. **build_dataset_embeddings.py** (NEW)
Main script that processes the entire dataset following notebook logic exactly.

**Key Features:**
- ✅ Tokenizes EACH NODE LABEL separately (not whole source code)
- ✅ Uses FIRST TOKEN embedding (not mean pooling)
- ✅ Node2Vec with exact notebook params: p=1, q=2, walk_length=100, num_walks=10
- ✅ Max token length: 128 (not 512)
- ✅ Outputs lists of tensors (notebook format)
- ✅ Includes metadata JSON with labels and splits

**Usage:**
```bash
# Test with first 5 samples
python build_dataset_embeddings.py --dataset agentic_dataset.json --limit 5 --device cpu

# Process full dataset
python build_dataset_embeddings.py --dataset agentic_dataset.json --device cuda:0
```

### 2. **EMBEDDING_PIPELINE.md** (NEW)
Complete documentation explaining:
- Critical differences between notebook and previous implementation
- Pipeline flow diagrams
- Usage instructions
- Verification steps
- Troubleshooting guide

### 3. **test_embeddings.py** (NEW)
Test script to verify embeddings match expected format.

**Usage:**
```bash
python test_embeddings.py
```

### 4. **src/dl_classifier/embeddings.py** (UPDATED)
Updated `JoernCodeT5Embedder` class documentation to explain notebook approach.
Now points users to `build_dataset_embeddings.py` for production use.

### 5. **.gitignore** (UPDATED)
Added patterns for:
- `embeddings_output/`
- `*.pt`, `*.pth`, `*.pkl`
- Temporary files

## Critical Differences Fixed

### 1. **CodeT5 Tokenization** (MOST IMPORTANT!)

**Before (WRONG):**
```python
# Tokenized entire source code once
inputs = tokenizer(source_code, max_length=512)
# Then replicated embedding for all nodes
embeddings = np.tile(embeddings, (num_nodes, 1))
```

**After (CORRECT - matches notebook):**
```python
# Extract node labels from graph
code_sequence = [str(node['label']) for node in nodes.values()]
# Tokenize EACH NODE LABEL separately
inputs = tokenizer(code_sequence, max_length=128)
# Result: unique embedding per node
```

### 2. **Embedding Extraction**

**Before:**
```python
embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
```

**After (matches notebook):**
```python
pooled_output = outputs.last_hidden_state[:, 0]  # First token
```

### 3. **Node2Vec Parameters**

**Before (flexible):**
- Parameters passed as arguments
- `max_length=512`

**After (fixed, matches notebook):**
- `p=1, q=2` (fixed)
- `walk_length=100` (not 20)
- `num_walks=10`
- `max_length=128`

### 4. **Output Format**

**Before (per-file):**
```
embeddings/
  ├── file1_graph_embeddings.pt
  ├── file1_token_embeddings.pt
  ├── file1_edge_index.pt
  ├── file2_graph_embeddings.pt
  ...
```

**After (batch, matches notebook):**
```
embeddings_output/
  ├── adj_matrices.pt                      # List of all edge indices
  ├── n2v_node_embeddings.pt              # List of all Node2Vec embeddings
  ├── aggregated_codet5_embeddings.pt     # List of all CodeT5 embeddings
  └── dataset_metadata.json               # Labels and splits
```

## Testing

Run the test script after generating embeddings:

```bash
# Generate test embeddings (first 5 samples)
python build_dataset_embeddings.py --dataset agentic_dataset.json --limit 5 --device cpu

# Verify format
python test_embeddings.py
```

Expected output:
```
✓ Found: adj_matrices.pt
✓ Found: n2v_node_embeddings.pt
✓ Found: aggregated_codet5_embeddings.pt
✓ Found: dataset_metadata.json

✓ Loaded embeddings successfully
  Number of samples: 5

Sample 0:
  ✓ Adjacency: torch.Size([2, 140])
  ✓ Node2Vec: torch.Size([129, 200])
  ✓ CodeT5: torch.Size([129, 768])
  ✓ Node counts match: 129
  ✓ Label: 1 (vulnerable)
  ✓ Split: train

✅ ALL CHECKS PASSED!
```

## Dataset Format

Your `agentic_dataset.json` is processed as:
- Uses `pre_patch` as source code
- Label `0` = benign, `1` = vulnerable
- Auto-assigns train/valid/test splits (70/15/15 by default)

## Next Steps

1. **Generate embeddings for full dataset:**
   ```bash
   python build_dataset_embeddings.py --dataset agentic_dataset.json --device cuda:0
   ```

2. **Load embeddings in training:**
   ```python
   import torch
   import json
   
   adj = torch.load('embeddings_output/adj_matrices.pt')
   n2v = torch.load('embeddings_output/n2v_node_embeddings.pt')
   codet5 = torch.load('embeddings_output/aggregated_codet5_embeddings.pt')
   
   with open('embeddings_output/dataset_metadata.json') as f:
       metadata = json.load(f)
   
   # Train your model...
   ```

3. **Verify with test script:**
   ```bash
   python test_embeddings.py
   ```

## Why This Matters

The per-node tokenization is **critical** for model accuracy:

- **Previous approach:** Same embedding copied to all nodes → no node-specific information
- **Notebook approach:** Each node gets its own embedding based on its label → preserves graph semantics

This is the main reason the notebook's model performs well - it captures the meaning of each individual graph node, not just the overall code.

## File Structure After Changes

```
covulpecker-kltn/
├── build_dataset_embeddings.py    # NEW: Main embedding script (matches notebook)
├── test_embeddings.py              # NEW: Verification script
├── EMBEDDING_PIPELINE.md           # NEW: Detailed documentation
├── precompute_embeddings.py        # OLD: Single-file processing (to be updated)
├── src/
│   └── dl_classifier/
│       └── embeddings.py           # UPDATED: Documentation
├── .gitignore                      # UPDATED: Added output patterns
├── embeddings/                     # Example embeddings
├── embeddings_output/              # NEW: Output directory (gitignored)
│   ├── adj_matrices.pt
│   ├── n2v_node_embeddings.pt
│   ├── aggregated_codet5_embeddings.pt
│   └── dataset_metadata.json
└── agentic_dataset.json            # Your dataset
```

## Summary

✅ **Embedding pipeline now exactly matches attention-covul.ipynb**
✅ **Per-node tokenization implemented correctly**
✅ **Output format matches notebook**
✅ **Documentation and tests included**
✅ **Ready for model training**

The key insight: **tokenize node labels, not source code** - this preserves the graph structure's semantic meaning in the embeddings.

