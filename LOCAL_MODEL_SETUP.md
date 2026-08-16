# 💻 Local Model & Hardware Acceleration Setup Guide

This guide details how to configure local models, embedding pipelines, and hardware acceleration (Metal Performance Shaders on Apple Silicon macOS) to run the Tax-Return-AI platform fully offline.

---

(TODO): What is the point of mentioning something that happens inside the code , without no user needed.
## 🚀 1. Hardware Acceleration (Apple Silicon macOS)

The embedding engine (`BAAI/bge-m3`) runs locally via PyTorch. To utilize the Apple Silicon GPU/Unified Memory (MPS):

1. **Verify Python Virtual Environment is Active**:
   ```bash
   source .venv/bin/activate
   ```
2. **Check MPS Compatibility**:
   Run this Python command to verify PyTorch detects your Mac GPU:
   ```bash
   python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
   ```
   *Expected Output: `MPS Available: True`*

*Note: The chunking pipeline in `backend/ingestion/ingest.py` automatically falls back to `cpu` if MPS/CUDA is not available.*

---

(TODO): same as other, very low level value. I would drop it.
## 🧠 2. Embedding Model (`BAAI/bge-m3`)

The ingestion pipeline uses the **BGE-M3** multi-lingual model for Late Chunking token embeddings.

- **Automatic Download**:
  The first time you execute `python backend/ingestion/ingest.py`, PyTorch/HuggingFace will automatically download the BGE-M3 model files (approx. 2.2 GB) to your HuggingFace cache directory:
  `~/.cache/huggingface/hub/models--BAAI--bge-m3/`

- **Pre-downloading Model (Optional)**:
  To trigger the download manually before running the ingestion pipeline:
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
  ```

---

(TODO): I am using llama-server, maybe we should switch to llama-server?
I don't remember how I downloaded the model
llama-server \\n  -m ./gemma-4-local/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \\n  -md ./gemma-4-local/mtp-gemma-4-12B-it.gguf \\n  --spec-type draft-mtp \\n  --spec-draft-n-max 2 \\n  -ngl 999 \\n  -fa on \\n  -fit off \\n  --no-kv-unified \\n  --port 9000 \\n-c 65536 -np 2

## 🤖 3. Local Voter LLMs (Ollama Setup)

The consensus pipeline runs three separate LLM voter instances. To run these fully locally:

### 1. Install Ollama
Download and install the macOS application from [ollama.com](https://ollama.com).

### 2. Pull the Models
Pull the target models (e.g. `gemma` or `llama3`) to your local machine:
```bash
# Pull Gemma (Gemma 2 IT or similar small models)
ollama pull gemma:2b

# Pull Llama 3
ollama pull llama3
```

### 3. Run Ollama Server
Ensure the Ollama desktop app is running, or start the service in your terminal:
```bash
ollama serve
```

(TODO): This should likely be the main part, it is up to the user how to acutally setup the server. We could add the server setup as an appendix for an example.
### 4. Configure .env file
Create a `.env` file in the root of the project to map the voter agents to your local Ollama endpoints:
```env
# Voter 1 (chronological dates focus) - Local Ollama Gemma
VOTER_1_BASE_URL="http://localhost:11434/v1"
VOTER_1_API_KEY="ollama"
VOTER_1_MODEL="gemma:2b"

# Voter 2 (arithmetic totals focus) - Local Ollama Llama3
VOTER_2_BASE_URL="http://localhost:11434/v1"
VOTER_2_API_KEY="ollama"
VOTER_2_MODEL="llama3"

# Voter 3 (asset classification focus) - Local Ollama Gemma
VOTER_3_BASE_URL="http://localhost:11434/v1"
VOTER_3_API_KEY="ollama"
VOTER_3_MODEL="gemma:2b"
```
(TODO): Again this is more for an appendix. I am not acutally using vllm anymore
### 5. Running via vLLM
If running local models using **vLLM** (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`):
1. Install and start your vLLM server:
   ```bash
   vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
   ```
2. By default, vLLM hosts its OpenAI-compatible endpoint at `http://localhost:8000`. Set your `.env` variables:
   ```env
   VOTER_1_BASE_URL="http://localhost:8000"
   VOTER_1_API_KEY="vllm" # Required non-empty placeholder string
   VOTER_1_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
   ```
   *Note: Our client automatically appends `/v1` internally if it is missing from the base URL (avoiding 404 Not Found errors on `/chat/completions`).*

---

(TODO): I would drop it. Maybe just ratin the troubleshooting in a FAQ
## 🗄️ 4. SQLite Vector Support (`sqlite-vec`)

We store both standard database fields and high-dimensional vectors in a single SQLite database using **`sqlite-vec`**.

- **Automatic loading**:
  The database manager (`backend/db_manager.py`) automatically downloads and loads the precompiled native library matches for your platform (macOS arm64) using the `sqlite_vec` python package bindings.
- **Troubleshooting**:
  If you get error logs regarding vector extensions:
  1. Make sure you have Xcode Command Line Tools installed:
     ```bash
     xcode-select --install
     ```
  2. Reinstall `sqlite-vec` package to trigger fresh compilation of bindings:
     ```bash
     pip install --force-reinstall sqlite-vec
     ```
