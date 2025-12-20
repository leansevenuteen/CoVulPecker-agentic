# CoVulPecker - Vulnerability Detection Pipeline

A multi-agent AI system for detecting and analyzing security vulnerabilities in C/C++ code using LangGraph and deep learning.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- An LLM API endpoint (self-hosted or cloud service)
- (Optional) PyTorch and torch_geometric for deep learning classifier

### Installation

1. **Clone the repository** (if not already done):
```bash
cd /Users/macbook/Documents/personal/covulpecker-kltn
```

2. **Activate virtual environment**:
```bash
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

For the deep learning classifier, also install:
```bash
# For CPU
pip install torch torchvision torchaudio

# For CUDA (check your CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install PyTorch Geometric
pip install torch_geometric torch_scatter
```

4. **Configure environment variables**:

Create a `.env` file in the project root:
```bash
# LLM API Configuration
LLM_API_BASE_URL=https://your-llm-api.com/v1
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_API_KEY=not-needed

# Deep Learning Classifier (optional)
DL_MODEL_PATH=models/best_model_fold1.pt
DL_DEVICE=auto
```

If you don't have a `.env` file, the system will use defaults from `src/config.py`.

## 🎯 Running the Project

### Basic Usage

Analyze the example file:
```bash
python main.py
```

Analyze a specific C/C++ file:
```bash
python main.py path/to/your/file.c
```

### Programmatic Usage

```python
from main import analyze_file, analyze_code

# Analyze a file
result = analyze_file("example.c", max_iterations=3)

# Analyze code string
source_code = """
void vulnerable_function(char *input) {
    char buffer[100];
    strcpy(buffer, input);  // Buffer overflow!
}
"""
result = analyze_code(source_code, max_iterations=3)

# Access results
print(f"Classification: {result['classification'].label}")
print(f"Vulnerabilities found: {len(result['detection'].vulnerabilities)}")
print(f"Confirmed: {result['verification'].vulnerability_confirmed}")
```

## 📊 Pipeline Stages

The system runs through 5 stages:

1. **Classifier** 🎯 - Classifies code as vulnerable/clean (uses CausalGAT DL model)
2. **Detection** 🔍 - Identifies specific vulnerabilities
3. **Reason** 🧠 - Explains vulnerabilities and suggests exploitation
4. **Critic** ⚖️ - Evaluates analysis quality (can loop back to Detection)
5. **Verification** ✅ - Confirms vulnerabilities with PoC

## 🤖 Deep Learning Classifier

### Current Status
- **Without trained model**: Auto-labels all code as "vulnerable" (placeholder mode)
- **With trained model**: Uses CausalGAT for real predictions

### Setting up the DL Classifier

1. **Train your model** (see `classifier.ipynb` for training code)

2. **Save the model weights**:
```python
# After training
torch.save(model.state_dict(), "models/best_model_fold1.pt")
```

3. **Configure the model path** in `.env`:
```bash
DL_MODEL_PATH=models/best_model_fold1.pt
```

4. **Implement code embeddings** (optional):
   - Edit `src/dl_classifier/embeddings.py`
   - Implement `JoernCodeT5Embedder` class
   - Use Joern for CPG extraction + CodeT5 for token embeddings

### Model Architecture
- **Type**: CausalGAT (Causal Attention Graph Neural Network)
- **Input**: Graph embeddings (200-dim) + Token embeddings (768-dim)
- **Fusion**: Cross-attention mechanism
- **Output**: Binary classification (vulnerable/clean)

## 📁 Project Structure

```
covulpecker-kltn/
├── main.py              # Entry point
├── example.c            # Sample C file for testing
├── requirements.txt     # Dependencies
├── classifier.ipynb     # DL model training notebook
├── .env                 # Environment configuration (create this)
├── logs/                # Conversation logs (auto-generated)
└── src/
    ├── agents/          # AI Agents
    │   ├── classifier.py   # Stage 1: Classification
    │   ├── detection.py    # Stage 2: Vulnerability Detection
    │   ├── reason.py       # Stage 3: Reasoning & PoC
    │   ├── critic.py       # Stage 4: Quality Evaluation
    │   └── verification.py # Stage 5: Final Verification
    ├── dl_classifier/   # Deep Learning Module
    │   ├── model.py        # CausalGAT implementation
    │   ├── embeddings.py   # Code → Graph → Embeddings
    │   └── service.py      # Classifier service
    ├── graph.py         # LangGraph workflow
    ├── models.py        # Data models (Pydantic)
    ├── state.py         # Graph state schema
    ├── config.py        # Configuration
    ├── llm.py           # LLM client
    ├── logger.py        # Conversation logger
    └── report.py        # Report generator
```

## 🔧 Configuration

Edit `src/config.py` or use environment variables:

### LLM Configuration
```python
LLM_API_BASE_URL = "https://your-api.com/v1"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MAX_CRITIC_ITERATIONS = 3  # Max feedback loops
```

### DL Classifier Configuration
```python
DL_MODEL_PATH = "models/best_model.pt"
DL_DEVICE = "cuda"  # or "cpu" or "auto"
DL_NUM_FEATURES = 200  # Graph embedding dimension
DL_TOKENS_DIM = 768    # CodeT5 embedding dimension
DL_FUSION_MODE = "cross_atten"  # Fusion strategy
```

## 📝 Output

The system generates:
1. **Console output**: Real-time progress and results
2. **Conversation logs**: 
   - `logs/conversation_TIMESTAMP.json` - Structured log
   - `logs/conversation_TIMESTAMP.md` - Human-readable log
3. **Final report**: Detailed vulnerability analysis

Example output:
```
🚀 Starting vulnerability detection pipeline...
📡 Using LLM API: https://your-api.com/v1
--------------------------------------------------
🔍 Calling DL Classifier (mode: placeholder)
   ⚠️ Model not loaded, using placeholder (auto-label: vulnerable)
✓ Completed: CLASSIFIER
✓ Completed: DETECTION
✓ Completed: REASON
✓ Completed: CRITIC
✓ Completed: VERIFICATION

==================== FINAL REPORT ====================
Classification: vulnerable (95.0% confidence)
Vulnerabilities Found: 3
- buffer_overflow (critical) at line 15
- format_string (high) at line 23
- use_after_free (high) at line 45
...
```

## 🛠️ Troubleshooting

### LLM API Issues
```
Error: Connection refused
→ Check if LLM API is running and LLM_API_BASE_URL is correct
```

### DL Classifier Not Loading
```
⚠️ PyTorch not available, cannot load model
→ Install PyTorch: pip install torch
```

```
⚠️ torch_geometric not available
→ Install: pip install torch_geometric torch_scatter
```

### Model Path Error
```
⚠️ Failed to load model: [Errno 2] No such file or directory
→ Check DL_MODEL_PATH in .env points to your trained model
```

## 🎓 Training Your Own Model

See `classifier.ipynb` for:
1. Data preparation (graph embeddings + token embeddings)
2. K-fold training with CausalGAT
3. Model evaluation and saving
4. Testing on adversarial samples

## 📚 References

- **CausalGAT**: https://github.com/yongduosui/CAL/tree/main
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

## 📧 Support

For issues or questions:
1. Check configuration in `.env` and `src/config.py`
2. Review logs in `logs/` directory
3. Ensure LLM API is accessible
4. Verify all dependencies are installed

---

**Status**: 
- ✅ Multi-agent pipeline: Ready
- ✅ LLM integration: Ready
- 🔄 DL Classifier: Integrated (needs trained model + embeddings)
- ✅ Conversation logging: Ready
- ✅ English prompts: Ready
