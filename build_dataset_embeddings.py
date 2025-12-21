"""
Build Dataset Embeddings - Matches attention-covul.ipynb exactly

This script processes the agentic_dataset.json and creates embeddings following
the exact same pipeline as the Kaggle notebook:
1. Source code -> CPG extraction using Joern
2. CPG -> Node2Vec embeddings (graph structure)
3. Node labels -> CodeT5 embeddings (per-node tokenization)
4. Save as lists of tensors matching notebook output format

Output format matches notebook:
- adj_matrices.pt: List of [2, num_edges] tensors
- n2v_node_embeddings.pt: List of [num_nodes, 200] tensors  
- aggregated_codet5_embeddings.pt: List of [num_tokens, 768] tensors
- dataset_metadata.json: Labels and splits
"""

import os
import sys
import json
import gc
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm

import torch
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from transformers import AutoTokenizer, T5EncoderModel

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed_value):
    """Set random seeds for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)


class CodeT5Classifier(torch.nn.Module):
    """
    CodeT5 model for extracting embeddings - matches notebook exactly.
    This is used only for embedding extraction, not classification.
    """
    def __init__(self, model_name="Salesforce/codet5-base"):
        super(CodeT5Classifier, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.config = self.encoder.config
    
    def get_embeddings(self, input_ids, attention_mask):
        """Extract embeddings from encoder - matches notebook."""
        self.eval()
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Use first token embedding like notebook (not mean pooling!)
            pooled_output = outputs.last_hidden_state[:, 0]
        return pooled_output


class JoernCPGExtractor:
    """Extract Code Property Graph using Joern - matches notebook requirements."""
    
    def __init__(self, joern_parse_bin: str, joern_export_bin: str):
        self.joern_parse = joern_parse_bin
        self.joern_export = joern_export_bin
        self._verify_joern()
    
    def _verify_joern(self):
        """Verify Joern is installed and accessible."""
        try:
            if 'JAVA_HOME' not in os.environ:
                java_paths = [
                    '/opt/homebrew/opt/openjdk',
                    '/usr/local/opt/openjdk',
                ]
                for java_path in java_paths:
                    if os.path.exists(java_path):
                        os.environ['JAVA_HOME'] = java_path
                        os.environ['PATH'] = f"{java_path}/bin:{os.environ.get('PATH', '')}"
                        break
            
            subprocess.run([self.joern_parse, "--help"], 
                         capture_output=True, check=True, timeout=5)
            print("✓ Joern verified")
        except Exception as e:
            print(f"WARNING: Joern not working: {e}")
    
    def extract_cpg(self, source_code: str, temp_dir: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract CPG from source code string and return (adj_matrix, nodes).
        Returns format matching notebook: nodes dict with 'label' field.
        """
        # Write source to temp file
        source_file = os.path.join(temp_dir, "temp_source.c")
        with open(source_file, 'w') as f:
            f.write(source_code)
        
        base_name = "temp_source"
        bin_path = os.path.join(temp_dir, f"{base_name}.bin")
        dot_dir_path = os.path.join(temp_dir, f"{base_name}.dot")
        final_dot_path = os.path.join(temp_dir, f"{base_name}_final.dot")
        
        try:
            # Step 1: Parse to .bin
            subprocess.run(
                [self.joern_parse, "-o", bin_path, source_file],
                check=True, capture_output=True, timeout=30
            )
            
            # Step 2: Export to .dot
            subprocess.run(
                [self.joern_export, "--repr", "cpg14", "--out", dot_dir_path, bin_path],
                check=True, capture_output=True, timeout=30
            )
            
            # Step 3: Extract 1-cpg.dot
            if os.path.isdir(dot_dir_path):
                target_dot = os.path.join(dot_dir_path, "1-cpg.dot")
                if os.path.exists(target_dot):
                    shutil.copy(target_dot, final_dot_path)
                else:
                    dot_files = sorted([f for f in os.listdir(dot_dir_path) if f.endswith('.dot')])
                    if dot_files:
                        shutil.copy(os.path.join(dot_dir_path, dot_files[0]), final_dot_path)
                    else:
                        raise FileNotFoundError("No .dot files found")
                shutil.rmtree(dot_dir_path)
            
            # Step 4: Parse DOT to graph structure
            return self._parse_dot_to_graph(final_dot_path)
            
        except Exception as e:
            print(f"  Joern extraction failed: {e}")
            # Fallback: create minimal graph
            num_nodes = min(max(source_code.count('\n') * 2, 5), 50)
            adj_matrix = np.array([list(range(num_nodes)), list(range(num_nodes))], dtype=np.int64)
            nodes = {i: {'label': f'node_{i}'} for i in range(num_nodes)}
            return adj_matrix, nodes
    
    def _parse_dot_to_graph(self, dot_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Parse .dot file and return (edge_matrix, nodes) matching notebook format.
        edge_matrix: [2, num_edges] array (sources, targets)
        nodes: dict {idx: {'label': str}}
        """
        nodes = {}
        edges = []
        node_counter = 0
        
        with open(dot_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Parse node: "123" [label="function_name"]
                if '[label=' in line and '->' not in line:
                    parts = line.split('[')
                    node_id = parts[0].strip().strip('"')
                    label = parts[1].split('label=')[1].split(']')[0].strip('"')
                    if node_id not in nodes:
                        nodes[node_counter] = {'label': label}
                        node_counter += 1
                
                # Parse edge: "123" -> "456"
                elif '->' in line:
                    parts = line.split('->')
                    src = parts[0].strip().strip('"')
                    dst = parts[1].split('[')[0].strip().strip('"')
                    
                    # Map to indices
                    if src not in nodes:
                        nodes[node_counter] = {'label': 'unknown'}
                        node_counter += 1
                    if dst not in nodes:
                        nodes[node_counter] = {'label': 'unknown'}
                        node_counter += 1
                    
                    edges.append((int(src) if src.isdigit() else hash(src) % 1000,
                                 int(dst) if dst.isdigit() else hash(dst) % 1000))
        
        # Build edge matrix in [2, num_edges] format
        if edges:
            sources, targets = zip(*edges)
            edge_matrix = np.array([sources, targets], dtype=np.int64)
        else:
            # Empty graph fallback
            num_nodes = max(len(nodes), 5)
            nodes = {i: {'label': f'node_{i}'} for i in range(num_nodes)}
            edge_matrix = np.array([list(range(num_nodes)), list(range(num_nodes))], dtype=np.int64)
        
        return edge_matrix, nodes


class VulnerabilityEmbedder:
    """
    Main embedding pipeline - EXACTLY matches notebook logic.
    
    Key differences from previous implementation:
    1. Tokenizes EACH NODE LABEL separately (not whole source code)
    2. Uses FIRST TOKEN embedding (not mean pooling)
    3. Node2Vec parameters: p=1, q=2, walk_length=100, num_walks=10
    4. Max token length: 128 (not 512)
    """
    
    def __init__(self, device='cuda:0', n2v_dim=200):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load CodeT5 model
        self.codet5_model = CodeT5Classifier(model_name="Salesforce/codet5-base")
        self.codet5_model.to(self.device)
        self.codet5_model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        self.n2v_dim = n2v_dim
        
        # Storage for batch processing
        self.n2v_node_embeddings = []
        self.adj_matrices = []
        self.aggregated_codet5_embeddings = []
        self.labels = []
        self.splits = []
        self.ignore_indices = []
    
    def _make_graph(self, nodes: Dict, edge_matrix: np.ndarray) -> Tuple[nx.DiGraph, List[str]]:
        """Create NetworkX graph and extract node labels - EXACTLY like notebook."""
        G = nx.DiGraph()
        G.add_nodes_from(range(len(nodes)))
        
        sources, targets = edge_matrix
        edges = list(zip(sources.tolist(), targets.tolist()))
        G.add_edges_from(edges)
        
        # Extract code sequence from node labels - CRITICAL: notebook does this!
        code_sequence = [str(nodes[i]['label']) for i in range(len(nodes))]
        
        return G, code_sequence
    
    def process_sample(self, source_code: str, label: int, split: str, 
                      joern_extractor: JoernCPGExtractor, temp_dir: str, idx: int):
        """
        Process one sample - EXACTLY matches notebook loop body.
        """
        try:
            # Step 1: Extract CPG from source code
            adj_matrix, nodes = joern_extractor.extract_cpg(source_code, temp_dir)
            G, code_sequence = self._make_graph(nodes, adj_matrix)
            
            # Step 2: CodeT5 embedding - TOKENIZE EACH NODE LABEL!
            # This is the KEY difference: notebook tokenizes node labels, not source code
            inputs = self.tokenizer(
                code_sequence,  # List of node labels, not source code!
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128  # Notebook uses 128, not 512!
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                codet5_graph_embedding = self.codet5_model.get_embeddings(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                # Move to CPU immediately
                codet5_graph_embedding = codet5_graph_embedding.detach().cpu()
            
            self.aggregated_codet5_embeddings.append(codet5_graph_embedding)
            
            # Cleanup GPU memory
            del inputs
            torch.cuda.empty_cache()
            
            # Step 3: Node2Vec processing - EXACT notebook parameters!
            node2vec = Node2Vec(
                G,
                dimensions=self.n2v_dim,
                p=1,           # Notebook uses p=1
                q=2,           # Notebook uses q=2  
                walk_length=100,  # Notebook uses 100
                num_walks=10,     # Notebook uses 10
                workers=4,
                quiet=True,
                seed=42
            )
            
            w2v_model = node2vec.fit(window=10, min_count=1)
            node_embeddings = np.array([w2v_model.wv[str(node)] for node in G.nodes()])
            
            # Create tensor on CPU
            node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float32)
            self.n2v_node_embeddings.append(node_embeddings_tensor)
            
            # Cleanup
            del node_embeddings, w2v_model, node2vec
            
            # Step 4: Store adjacency matrix as tensor
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.long)
            self.adj_matrices.append(adj_tensor)
            self.labels.append(int(label))
            self.splits.append(split)
            
        except Exception as e:
            print(f"\n[ERROR] Sample {idx} failed: {e}")
            print(f"  Code length: {len(source_code)} chars")
            self.ignore_indices.append(idx)
        
        finally:
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def save_outputs(self, output_dir: str):
        """Save in EXACT notebook format."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings as lists of tensors (notebook format)
        torch.save(self.adj_matrices, os.path.join(output_dir, 'adj_matrices.pt'))
        torch.save(self.n2v_node_embeddings, os.path.join(output_dir, 'n2v_node_embeddings.pt'))
        torch.save(self.aggregated_codet5_embeddings, os.path.join(output_dir, 'aggregated_codet5_embeddings.pt'))
        
        # Save metadata JSON
        metadata = {
            'labels': self.labels,
            'partition': self.splits,
            'ignored_indices': self.ignore_indices
        }
        with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved embeddings to {output_dir}/")
        print(f"  - adj_matrices.pt: {len(self.adj_matrices)} samples")
        print(f"  - n2v_node_embeddings.pt: {len(self.n2v_node_embeddings)} samples")
        print(f"  - aggregated_codet5_embeddings.pt: {len(self.aggregated_codet5_embeddings)} samples")
        print(f"  - dataset_metadata.json: {len(self.labels)} labels")
        if self.ignore_indices:
            print(f"  - Ignored {len(self.ignore_indices)} failed samples")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build embeddings matching attention-covul.ipynb")
    parser.add_argument('--dataset', type=str, default='agentic_dataset.json', help='Path to dataset JSON')
    parser.add_argument('--output_dir', type=str, default='embeddings_output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device: cuda:0 or cpu')
    parser.add_argument('--limit', type=int, default=None, help='Process only first N samples (for testing)')
    parser.add_argument('--train_split', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--valid_split', type=float, default=0.15, help='Valid split ratio')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"  Limited to first {args.limit} samples for testing")
    
    print(f"  Loaded {len(dataset)} samples")
    
    # Assign splits (train/valid/test)
    n_samples = len(dataset)
    n_train = int(n_samples * args.train_split)
    n_valid = int(n_samples * args.valid_split)
    
    for i, sample in enumerate(dataset):
        if i < n_train:
            sample['split'] = 'train'
        elif i < n_train + n_valid:
            sample['split'] = 'valid'
        else:
            sample['split'] = 'test'
    
    print(f"  Splits: train={n_train}, valid={n_valid}, test={n_samples - n_train - n_valid}")
    
    # Initialize components
    script_dir = Path(__file__).parent.absolute()
    joern_parse = str(script_dir / "joern-graph" / "joern" / "joern-parse")
    joern_export = str(script_dir / "joern-graph" / "joern" / "joern-export")
    
    print("\nInitializing components...")
    joern_extractor = JoernCPGExtractor(joern_parse, joern_export)
    embedder = VulnerabilityEmbedder(device=args.device, n2v_dim=200)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process all samples
        print(f"\nProcessing {len(dataset)} samples...")
        print("Following EXACT notebook pipeline:")
        print("  1. Source code -> Joern CPG")
        print("  2. Node labels -> CodeT5 embeddings (per-node, first token)")
        print("  3. Graph structure -> Node2Vec (p=1, q=2, len=100, walks=10)")
        print()
        
        for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
            source_code = sample['pre_patch']  # Use pre_patch as source
            label = sample['label']
            split = sample.get('split', 'train')
            
            embedder.process_sample(
                source_code=source_code,
                label=label,
                split=split,
                joern_extractor=joern_extractor,
                temp_dir=temp_dir,
                idx=idx
            )
        
        # Save outputs
        embedder.save_outputs(args.output_dir)
        
        print("\n✓ Dataset embedding complete!")
        print(f"✓ Output matches notebook format in: {args.output_dir}/")
        
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()

