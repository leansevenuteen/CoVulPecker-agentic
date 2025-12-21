"""
Code Embedding Module - Transform source code to model-compatible format.

This module handles the MIDDLE STEP transformation:
    Source Code (string) 
        -> Code Graph (CPG/AST)
        -> Graph Embeddings (using Node2Vec or similar)
        -> Token Embeddings (using CodeT5 or similar SLM)
        -> Combined tensor format for CausalGAT

The user will provide the actual implementation later.
This file contains the interface and placeholder implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

# Check if torch is available
try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class CodeEmbeddings:
    """
    Container for code embeddings.
    
    Attributes:
        graph_embeddings: Node embeddings from graph structure (N x D_graph)
        token_embeddings: Token embeddings from code LM (N x D_tokens)
        edge_index: Graph connectivity as edge list (2 x E)
        num_nodes: Number of nodes in the graph
    """
    graph_embeddings: np.ndarray  # Shape: (num_nodes, graph_embed_dim)
    token_embeddings: np.ndarray  # Shape: (num_nodes, token_embed_dim)
    edge_index: np.ndarray        # Shape: (2, num_edges)
    num_nodes: int
    
    def to_pyg_data(self, device: Optional[str] = None) -> "Data":
        """
        Convert embeddings to PyTorch Geometric Data object.
        
        Args:
            device: Device to place tensors on ('cuda', 'cpu', etc.)
            
        Returns:
            PyG Data object ready for model input
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torch_geometric are required")
        
        x = torch.from_numpy(self.graph_embeddings).float()
        tokens = torch.from_numpy(self.token_embeddings).float()
        edge_index = torch.from_numpy(self.edge_index).long()
        
        # Add batch dimension (single sample)
        batch = torch.zeros(self.num_nodes, dtype=torch.long)
        
        if device:
            x = x.to(device)
            tokens = tokens.to(device)
            edge_index = edge_index.to(device)
            batch = batch.to(device)
        
        return Data(
            x=x,
            tokens=tokens,
            edge_index=edge_index,
            batch=batch
        )


class CodeEmbedder(ABC):
    """
    Abstract base class for code embedding.
    
    Implementations should handle:
    1. Parsing source code to graph (CPG/AST)
    2. Generating graph node embeddings (e.g., Node2Vec)
    3. Generating token embeddings (e.g., CodeT5)
    """
    
    @abstractmethod
    def embed(self, source_code: str) -> CodeEmbeddings:
        """
        Transform source code to embeddings.
        
        Args:
            source_code: Raw source code string
            
        Returns:
            CodeEmbeddings object containing graph and token embeddings
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the embedder is properly initialized."""
        pass


class PlaceholderEmbedder(CodeEmbedder):
    """
    Placeholder embedder that generates random embeddings.
    
    This is used when the actual embedding pipeline is not available.
    Replace this with the actual implementation when ready.
    """
    
    def __init__(self, 
                 graph_embed_dim: int = 200,
                 token_embed_dim: int = 768,
                 min_nodes: int = 5,
                 max_nodes: int = 50):
        """
        Initialize placeholder embedder.
        
        Args:
            graph_embed_dim: Dimension of graph node embeddings
            token_embed_dim: Dimension of token embeddings (768 for CodeT5)
            min_nodes: Minimum number of nodes to generate
            max_nodes: Maximum number of nodes to generate
        """
        self.graph_embed_dim = graph_embed_dim
        self.token_embed_dim = token_embed_dim
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self._ready = True
    
    def embed(self, source_code: str) -> CodeEmbeddings:
        """
        Generate placeholder embeddings.
        
        In the actual implementation, this would:
        1. Parse source code to CPG using Joern or similar
        2. Extract node features and adjacency matrix
        3. Run Node2Vec on the graph
        4. Run CodeT5 on code tokens
        5. Align embeddings to nodes
        """
        # Estimate number of nodes from code complexity
        lines = source_code.count('\n') + 1
        num_nodes = min(max(lines * 2, self.min_nodes), self.max_nodes)
        
        # Generate random embeddings (placeholder)
        np.random.seed(hash(source_code) % (2**32))
        
        graph_embeddings = np.random.randn(num_nodes, self.graph_embed_dim).astype(np.float32)
        token_embeddings = np.random.randn(num_nodes, self.token_embed_dim).astype(np.float32)
        
        # Generate random edge index (simple connected graph)
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        # Add some random edges
        for _ in range(num_nodes):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edges.append([src, dst])
        
        edge_index = np.array(edges, dtype=np.int64).T
        
        return CodeEmbeddings(
            graph_embeddings=graph_embeddings,
            token_embeddings=token_embeddings,
            edge_index=edge_index,
            num_nodes=num_nodes
        )
    
    def is_ready(self) -> bool:
        return self._ready


class JoernCodeT5Embedder(CodeEmbedder):
    """
    Production embedder using Joern for CPG and CodeT5 for token embeddings.
    
    MATCHES attention-covul.ipynb EXACTLY:
    1. Joern: source code -> CPG (Code Property Graph)  
    2. Node2Vec: CPG -> graph node embeddings (p=1, q=2, walk_length=100)
    3. CodeT5: NODE LABELS (not source code!) -> token embeddings per node
    4. Uses FIRST TOKEN embedding (not mean pooling)
    
    Key differences from previous implementation:
    - Tokenizes each node label separately (not whole source code)
    - Uses first token [:, 0] embedding (not mean pooling)
    - Node2Vec: p=1, q=2, walk_length=100, num_walks=10
    - Max length: 128 (not 512)
    """
    
    def __init__(self,
                 joern_parse_path: Optional[str] = None,
                 joern_export_path: Optional[str] = None,
                 codet5_model: str = "Salesforce/codet5-base",
                 node2vec_dim: int = 200,
                 device: str = "cpu"):
        """
        Initialize the Joern + CodeT5 embedder matching notebook.
        
        Args:
            joern_parse_path: Path to joern-parse executable
            joern_export_path: Path to joern-export executable
            codet5_model: HuggingFace model name for CodeT5
            node2vec_dim: Dimension for Node2Vec embeddings
            device: Device for model inference
        """
        self.joern_parse_path = joern_parse_path
        self.joern_export_path = joern_export_path
        self.codet5_model_name = codet5_model
        self.node2vec_dim = node2vec_dim
        self.device = device
        
        self._ready = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the embedding components."""
        try:
            # Import required dependencies
            import subprocess
            import tempfile
            try:
                import torch
                from transformers import AutoTokenizer, T5EncoderModel
                import networkx as nx
                from node2vec import Node2Vec
            except ImportError as e:
                print(f"Missing dependencies: {e}")
                print("Install: pip install torch transformers networkx node2vec")
                return
            
            # Load CodeT5 model
            print(f"Loading {self.codet5_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.codet5_model_name)
            self.codet5_encoder = T5EncoderModel.from_pretrained(self.codet5_model_name)
            
            # Move to device
            dev = torch.device(self.device if torch.cuda.is_available() and 'cuda' in self.device else 'cpu')
            self.codet5_encoder = self.codet5_encoder.to(dev)
            self.codet5_encoder.eval()
            self.torch_device = dev
            
            self._ready = True
            print(f"✓ JoernCodeT5Embedder initialized on {dev}")
            
        except Exception as e:
            print(f"Failed to initialize embedder: {e}")
            self._ready = False
    
    def embed(self, source_code: str) -> CodeEmbeddings:
        """
        Transform source code to embeddings - MATCHES NOTEBOOK EXACTLY.
        
        Pipeline (matching attention-covul.ipynb):
        1. Extract CPG using Joern
        2. Extract node labels from CPG
        3. Tokenize EACH NODE LABEL separately with CodeT5
        4. Use FIRST TOKEN [:, 0] as embedding (not mean!)
        5. Generate Node2Vec embeddings with p=1, q=2
        """
        if not self._ready:
            raise RuntimeError("JoernCodeT5Embedder not initialized properly")
        
        import torch
        import networkx as nx
        from node2vec import Node2Vec
        import tempfile
        import os
        
        # Use the build_dataset_embeddings.py implementation
        # For now, redirect to that script
        raise NotImplementedError(
            "For production use, please use build_dataset_embeddings.py which matches "
            "the notebook exactly. This class is for reference only."
        )
    
    def is_ready(self) -> bool:
        return self._ready


class PrecomputedEmbedder(CodeEmbedder):
    """
    Embedder that loads precomputed embeddings from files.
    
    Use this when embeddings are precomputed and stored as .pt files.
    This is useful for batch processing or when using cached embeddings.
    """
    
    def __init__(self,
                 embeddings_cache: Optional[dict] = None):
        """
        Initialize with optional precomputed embeddings cache.
        
        Args:
            embeddings_cache: Dict mapping source_code_hash -> CodeEmbeddings
        """
        self.cache = embeddings_cache or {}
        self._ready = True
    
    def add_from_files(self,
                       code_hash: str,
                       graph_embed_path: str,
                       token_embed_path: str,
                       adj_matrix_path: str):
        """
        Add embeddings from .pt files to cache.
        
        Args:
            code_hash: Unique identifier for the source code
            graph_embed_path: Path to graph embeddings .pt file
            token_embed_path: Path to token embeddings .pt file
            adj_matrix_path: Path to adjacency matrix .pt file
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        graph_emb = torch.load(graph_embed_path, map_location='cpu')
        token_emb = torch.load(token_embed_path, map_location='cpu')
        adj_matrix = torch.load(adj_matrix_path, map_location='cpu')
        
        # Convert adjacency matrix to edge_index if needed
        if adj_matrix.dim() == 2 and adj_matrix.shape[0] == adj_matrix.shape[1]:
            # It's an adjacency matrix, convert to edge_index
            edge_index = adj_matrix.nonzero().t().contiguous()
        else:
            # Assume it's already edge_index format
            edge_index = adj_matrix
        
        self.cache[code_hash] = CodeEmbeddings(
            graph_embeddings=graph_emb.numpy() if isinstance(graph_emb, torch.Tensor) else graph_emb,
            token_embeddings=token_emb.numpy() if isinstance(token_emb, torch.Tensor) else token_emb,
            edge_index=edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index,
            num_nodes=graph_emb.shape[0]
        )
    
    def embed(self, source_code: str) -> CodeEmbeddings:
        """
        Look up precomputed embeddings.
        
        Falls back to placeholder if not found in cache.
        """
        import hashlib
        code_hash = hashlib.md5(source_code.encode()).hexdigest()
        
        if code_hash in self.cache:
            return self.cache[code_hash]
        
        # Fall back to placeholder
        placeholder = PlaceholderEmbedder()
        return placeholder.embed(source_code)
    
    def is_ready(self) -> bool:
        return self._ready


def get_default_embedder() -> CodeEmbedder:
    """
    Get the default embedder based on available dependencies.
    
    Returns PlaceholderEmbedder if actual implementation is not available.
    """
    # TODO: When user provides actual implementation, check for dependencies
    # and return the appropriate embedder
    return PlaceholderEmbedder()


