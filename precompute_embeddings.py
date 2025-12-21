"""
Precompute Embeddings Script

This script generates embeddings for source code files using:
1. Joern for CPG extraction
2. Node2Vec for graph embeddings
3. CodeT5/CodeLM for token embeddings

Usage:
    python precompute_embeddings.py --input source.c --output embeddings/
    python precompute_embeddings.py --input_dir samples/ --output_dir embeddings/
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import numpy as np
from tqdm import tqdm


class JoernCPGExtractor:
    """Extract Code Property Graph using Joern."""
    
    def __init__(self, joern_parse_bin: str = "joern-parse", joern_export_bin: str = "joern-export"):
        """
        Initialize Joern extractor.
        
        Args:
            joern_parse_bin: Path to joern-parse executable
            joern_export_bin: Path to joern-export executable
        """
        self.joern_parse = joern_parse_bin
        self.joern_export = joern_export_bin
        
        # Verify Joern is installed
        self._verify_joern()
    
    def _verify_joern(self):
        """Verify Joern is installed and accessible."""
        try:
            # Set up Java environment if not already set
            if 'JAVA_HOME' not in os.environ:
                # Try common Homebrew locations
                java_paths = [
                    '/opt/homebrew/opt/openjdk',
                    '/usr/local/opt/openjdk',
                    '/opt/homebrew/opt/openjdk@19',
                    '/usr/local/opt/openjdk@19'
                ]
                for java_path in java_paths:
                    if os.path.exists(java_path):
                        os.environ['JAVA_HOME'] = java_path
                        os.environ['PATH'] = f"{java_path}/bin:{os.environ.get('PATH', '')}"
                        break
            
            # Test with --help instead of --version
            result = subprocess.run([self.joern_parse, "--help"], 
                         capture_output=True, check=True, timeout=5)
            print("Joern found and verified")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print("WARNING: Joern not found or not working. Install from: https://joern.io/")
            print(f"Error details: {e}")
            print("The script will continue but CPG extraction will fail.")
    
    def extract_cpg(self, source_file: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Extract CPG from source file using Joern.
        
        Args:
            source_file: Path to source code file
            output_dir: Directory to save outputs (uses temp if None)
            
        Returns:
            Tuple of (bin_path, dot_path)
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = Path(source_file).stem
        bin_path = os.path.join(output_dir, f"{base_name}.bin")
        dot_dir_path = os.path.join(output_dir, f"{base_name}.dot")
        final_dot_path = os.path.join(output_dir, f"{base_name}_final.dot")
        
        try:
            # Step 1: Parse source to .bin
            print(f"  Parsing {source_file}...")
            subprocess.run(
                [self.joern_parse, "-o", bin_path, source_file],
                check=True,
                capture_output=True,
                timeout=30
            )
            
            # Step 2: Export CPG to .dot (creates a directory with multiple .dot files)
            print(f"  Exporting CPG to {dot_dir_path}...")
            subprocess.run(
                [self.joern_export, "--repr", "cpg14", "--out", dot_dir_path, bin_path],
                check=True,
                capture_output=True,
                timeout=30
            )
            
            # Step 3: Extract the 1-cpg.dot file and cleanup
            if os.path.isdir(dot_dir_path):
                target_dot = os.path.join(dot_dir_path, "1-cpg.dot")
                if os.path.exists(target_dot):
                    # Copy the 1-cpg.dot to final location
                    shutil.copy(target_dot, final_dot_path)
                    print(f"  Extracted 1-cpg.dot from {dot_dir_path}")
                else:
                    # Fallback: try to find any .dot file
                    dot_files = sorted([f for f in os.listdir(dot_dir_path) if f.endswith('.dot')])
                    if dot_files:
                        first_dot = os.path.join(dot_dir_path, dot_files[0])
                        shutil.copy(first_dot, final_dot_path)
                        print(f"  Using {dot_files[0]} as CPG file")
                    else:
                        raise FileNotFoundError(f"No .dot files found in {dot_dir_path}")
                
                # Cleanup: remove the directory
                shutil.rmtree(dot_dir_path)
            else:
                # If it's a file (shouldn't happen), just use it
                final_dot_path = dot_dir_path
            
            return bin_path, final_dot_path
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Joern failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        except subprocess.TimeoutExpired:
            print("ERROR: Joern timed out")
            raise
    
    def parse_dot_to_graph(self, dot_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Parse .dot file to extract graph structure.
        
        Args:
            dot_path: Path to .dot file
            
        Returns:
            Tuple of (adjacency_matrix, node_info)
        """
        # Simple parser for DOT format
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
                        nodes[node_id] = {'idx': node_counter, 'label': label}
                        node_counter += 1
                
                # Parse edge: "123" -> "456"
                elif '->' in line:
                    parts = line.split('->')
                    src = parts[0].strip().strip('"')
                    dst = parts[1].split('[')[0].strip().strip('"')
                    
                    # Create nodes if they don't exist
                    if src not in nodes:
                        nodes[src] = {'idx': node_counter, 'label': 'unknown'}
                        node_counter += 1
                    if dst not in nodes:
                        nodes[dst] = {'idx': node_counter, 'label': 'unknown'}
                        node_counter += 1
                    
                    edges.append((nodes[src]['idx'], nodes[dst]['idx']))
        
        # Build adjacency matrix
        num_nodes = len(nodes)
        if num_nodes == 0:
            # Return minimal graph if parsing failed
            num_nodes = 5
            adj_matrix = np.eye(num_nodes, dtype=np.int64)
            nodes = {str(i): {'idx': i, 'label': 'node'} for i in range(num_nodes)}
        else:
            adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)
            for src, dst in edges:
                adj_matrix[src, dst] = 1
        
        return adj_matrix, nodes


class Node2VecEmbedder:
    """Generate graph embeddings using Node2Vec."""
    
    def __init__(self, embedding_dim: int = 200, device: str = "cpu"):
        """
        Initialize Node2Vec embedder.
        
        Args:
            embedding_dim: Dimension of node embeddings
            device: Device for computation
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_real_node2vec = self._check_node2vec_available()
    
    def _check_node2vec_available(self) -> bool:
        """Check if node2vec and networkx are available."""
        try:
            import networkx as nx
            from node2vec import Node2Vec
            return True
        except ImportError:
            print("WARNING: node2vec or networkx not installed.")
            print("Install with: pip install node2vec networkx")
            print("Falling back to random embeddings.")
            return False
    
    def embed(self, adj_matrix: np.ndarray, num_walks: int = 10, walk_length: int = 20, 
              p: float = 1.0, q: float = 1.0, workers: int = 4) -> np.ndarray:
        """
        Generate node embeddings from adjacency matrix using Node2Vec.
        
        Args:
            adj_matrix: Adjacency matrix [N, N]
            num_walks: Number of random walks per node
            walk_length: Length of each walk
            p: Return parameter (controls likelihood of returning to previous node)
            q: In-out parameter (controls BFS vs DFS)
            workers: Number of parallel workers
            
        Returns:
            Node embeddings [N, embedding_dim]
        """
        num_nodes = adj_matrix.shape[0]
        
        if self.use_real_node2vec:
            try:
                import networkx as nx
                from node2vec import Node2Vec
                from gensim.models import Word2Vec
                
                # Convert adjacency matrix to NetworkX graph
                G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
                
                # Check if graph is empty
                if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                    print("  WARNING: Empty graph, using random embeddings")
                    return self._fallback_embeddings(num_nodes)
                
                # Run Node2Vec
                print(f"  Running Node2Vec on graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges...")
                node2vec = Node2Vec(
                    G, 
                    dimensions=self.embedding_dim,
                    walk_length=walk_length,
                    num_walks=num_walks,
                    p=p,
                    q=q,
                    workers=workers,
                    quiet=True
                )
                
                # Train Skip-gram model
                model = node2vec.fit(
                    window=10,
                    min_count=1,
                    batch_words=4,
                    epochs=5,
                    sg=1  # Skip-gram
                )
                
                # Extract embeddings for all nodes
                embeddings = np.zeros((num_nodes, self.embedding_dim), dtype=np.float32)
                for node_id in range(num_nodes):
                    if str(node_id) in model.wv:
                        embeddings[node_id] = model.wv[str(node_id)]
                    else:
                        # Node not in vocabulary, use random
                        embeddings[node_id] = np.random.randn(self.embedding_dim)
                
                # Normalize embeddings
                embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
                
                print(f"  Generated Node2Vec embeddings: {embeddings.shape}")
                return embeddings
                
            except Exception as e:
                print(f"  WARNING: Node2Vec failed: {e}")
                print("  Falling back to random embeddings")
                return self._fallback_embeddings(num_nodes)
        else:
            return self._fallback_embeddings(num_nodes)
    
    def _fallback_embeddings(self, num_nodes: int) -> np.ndarray:
        """Generate random fallback embeddings."""
        embeddings = np.random.randn(num_nodes, self.embedding_dim).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        print(f"  Generated random embeddings (fallback): {embeddings.shape}")
        return embeddings


class CodeT5Embedder:
    """Generate token embeddings using CodeT5 or similar language model."""
    
    def __init__(self, model_name: str = "Salesforce/codet5-base", device: str = "cpu"):
        """
        Initialize CodeT5 embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device for computation
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load CodeT5 model and tokenizer."""
        try:
            from transformers import AutoTokenizer, T5EncoderModel
            import torch
            
            print(f"  Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            # Use T5EncoderModel to get encoder-only outputs
            self.model = T5EncoderModel.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Move to device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print(f"  CodeT5 loaded successfully on GPU")
            else:
                self.model = self.model.to('cpu')
                print(f"  CodeT5 loaded successfully on CPU")
            
            self.model.eval()
            self.model_loaded = True
            
        except ImportError:
            print("WARNING: transformers not installed. Install: pip install transformers torch")
            print("Using random embeddings as fallback.")
            self.model_loaded = False
        except Exception as e:
            print(f"WARNING: Failed to load CodeT5: {e}")
            print("Using random embeddings as fallback.")
            self.model_loaded = False
    
    def embed(self, source_code: str, num_nodes: int, max_length: int = 512) -> np.ndarray:
        """
        Generate token embeddings from source code.
        
        Args:
            source_code: Source code string
            num_nodes: Number of nodes (to match graph structure)
            max_length: Maximum sequence length
            
        Returns:
            Token embeddings [num_nodes, 768]
        """
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            # Fallback to random embeddings
            embeddings = np.random.randn(num_nodes, 768).astype(np.float32)
            print(f"  Generated random token embeddings (fallback): {embeddings.shape}")
            return embeddings
        
        try:
            import torch
            
            # Tokenize and encode
            inputs = self.tokenizer(
                source_code,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Move to device
            if self.device == 'cuda' and torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            else:
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state and pool (mean pooling across sequence)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            
            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
            
            # Expand to match num_nodes (replicate the code embedding for each node)
            embeddings = np.tile(embeddings, (num_nodes, 1))[:num_nodes]
            
            print(f"  Generated CodeT5 embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"WARNING: CodeT5 embedding failed: {e}")
            import traceback
            traceback.print_exc()
            embeddings = np.random.randn(num_nodes, 768).astype(np.float32)
            return embeddings


def precompute_single_file(
    source_file: str,
    output_dir: str,
    joern_extractor: JoernCPGExtractor,
    node2vec_embedder: Node2VecEmbedder,
    codet5_embedder: CodeT5Embedder,
    temp_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Precompute embeddings for a single source file.
    
    Args:
        source_file: Path to source code file
        output_dir: Directory to save embeddings
        joern_extractor: Joern CPG extractor
        node2vec_embedder: Node2Vec embedder
        codet5_embedder: CodeT5 embedder
        temp_dir: Temporary directory for intermediate files
        
    Returns:
        Dictionary with paths to generated files
    """
    base_name = Path(source_file).stem
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing: {source_file}")
    
    # Step 1: Extract CPG using Joern
    try:
        bin_path, dot_path = joern_extractor.extract_cpg(source_file, temp_dir)
        adj_matrix, node_info = joern_extractor.parse_dot_to_graph(dot_path)
    except Exception as e:
        print(f"  WARNING: CPG extraction failed: {e}")
        print("  Using fallback graph structure")
        # Fallback: create minimal graph
        with open(source_file, 'r') as f:
            source_code = f.read()
        num_nodes = min(max(source_code.count('\n') * 2, 5), 50)
        adj_matrix = np.eye(num_nodes, dtype=np.int64)
        node_info = {str(i): {'idx': i, 'label': 'node'} for i in range(num_nodes)}
    
    num_nodes = adj_matrix.shape[0]
    print(f"  Graph: {num_nodes} nodes, {adj_matrix.sum()} edges")
    
    # Step 2: Generate graph embeddings using Node2Vec
    graph_embeddings = node2vec_embedder.embed(adj_matrix)
    
    # Step 3: Read source code and generate token embeddings
    with open(source_file, 'r') as f:
        source_code = f.read()
    
    token_embeddings = codet5_embedder.embed(source_code, num_nodes)
    
    # Step 4: Convert adjacency matrix to edge_index format
    edge_index = np.array(np.where(adj_matrix > 0), dtype=np.int64)
    
    # Step 5: Save all embeddings
    graph_path = os.path.join(output_dir, f"{base_name}_graph_embeddings.pt")
    token_path = os.path.join(output_dir, f"{base_name}_token_embeddings.pt")
    adj_path = os.path.join(output_dir, f"{base_name}_edge_index.pt")
    
    torch.save(torch.from_numpy(graph_embeddings), graph_path)
    torch.save(torch.from_numpy(token_embeddings), token_path)
    torch.save(torch.from_numpy(edge_index), adj_path)
    
    print(f"  Saved embeddings to {output_dir}/")
    
    return {
        'graph_embeddings': graph_path,
        'token_embeddings': token_path,
        'edge_index': adj_path,
        'num_nodes': num_nodes
    }


def main():
    # Get the script directory to find Joern installation
    script_dir = Path(__file__).parent.absolute()
    default_joern_parse = str(script_dir / "joern-graph" / "joern" / "joern-parse")
    default_joern_export = str(script_dir / "joern-graph" / "joern" / "joern-export")
    
    parser = argparse.ArgumentParser(description="Precompute embeddings for vulnerability detection")
    parser.add_argument('--input', type=str, help='Single source file to process')
    parser.add_argument('--input_dir', type=str, help='Directory with source files to process')
    parser.add_argument('--output_dir', type=str, default='embeddings', help='Output directory for embeddings')
    parser.add_argument('--joern_parse', type=str, default=default_joern_parse, help='Path to joern-parse')
    parser.add_argument('--joern_export', type=str, default=default_joern_export, help='Path to joern-export')
    parser.add_argument('--model', type=str, default='Salesforce/codet5-base', help='CodeT5 model name')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--no_codet5', action='store_true', help='Skip CodeT5, use random embeddings')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input and not args.input_dir:
        print("ERROR: Must specify --input or --input_dir")
        sys.exit(1)
    
    # Initialize extractors/embedders
    print("Initializing extractors and embedders...")
    joern_extractor = JoernCPGExtractor(args.joern_parse, args.joern_export)
    node2vec_embedder = Node2VecEmbedder(embedding_dim=200, device=args.device)
    
    if args.no_codet5:
        codet5_embedder = None
    else:
        codet5_embedder = CodeT5Embedder(model_name=args.model, device=args.device)
    
    # If no CodeT5, use dummy
    if codet5_embedder is None:
        class DummyEmbedder:
            def embed(self, source_code, num_nodes):
                return np.random.randn(num_nodes, 768).astype(np.float32)
        codet5_embedder = DummyEmbedder()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process files
        if args.input:
            # Single file
            results = precompute_single_file(
                args.input,
                args.output_dir,
                joern_extractor,
                node2vec_embedder,
                codet5_embedder,
                temp_dir
            )
            print(f"\nSuccess! Embeddings saved to: {args.output_dir}/")
            
        elif args.input_dir:
            # Directory of files
            source_files = list(Path(args.input_dir).glob('*.c')) + list(Path(args.input_dir).glob('*.cpp'))
            
            if not source_files:
                print(f"No .c or .cpp files found in {args.input_dir}")
                sys.exit(1)
            
            print(f"Found {len(source_files)} source files")
            
            all_results = []
            for source_file in tqdm(source_files, desc="Processing files"):
                try:
                    results = precompute_single_file(
                        str(source_file),
                        args.output_dir,
                        joern_extractor,
                        node2vec_embedder,
                        codet5_embedder,
                        temp_dir
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"ERROR processing {source_file}: {e}")
            
            print(f"\nSuccess! Processed {len(all_results)}/{len(source_files)} files")
            print(f"Embeddings saved to: {args.output_dir}/")
    
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()

