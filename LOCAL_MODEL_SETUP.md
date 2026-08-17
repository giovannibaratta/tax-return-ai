# 💻 Local Model & Endpoint Setup Guide

This guide details how to configure local OpenAI-compatible endpoints for the Tax-Return-AI platform.

---

## 🤖 1. Configuring Local LLM Endpoints (`.env`)

Tax-Return-AI connects to any OpenAI-compatible API endpoint for voter LLMs, deliberation agents, and chat assistants. Configure the target endpoints in your `.env` file:

```env
# Multi-Voter Consensus Endpoints (Local or Remote)
VOTER_1_BASE_URL="http://localhost:9000/v1"
VOTER_1_API_KEY="local-llama"
VOTER_1_MODEL="gemma-4-12B-it"

VOTER_2_BASE_URL="http://localhost:9000/v1"
VOTER_2_API_KEY="local-llama"
VOTER_2_MODEL="gemma-4-12B-it"

VOTER_3_BASE_URL="http://localhost:9000/v1"
VOTER_3_API_KEY="local-llama"
VOTER_3_MODEL="gemma-4-12B-it"

# Chat / Deliberation LLM Configuration
CHAT_BASE_URL="http://localhost:9000/v1"
CHAT_API_KEY="local-llama"
CHAT_MODEL="gemma-4-12B-it"
```

*Note: The runner client automatically appends `/v1` internally if omitted from the base URL.*

---

## 🛠️ Appendix: Example Local Server Setups

You can use any local inference server of your choice. Below are popular examples:

### Option A: `llama-server` (llama.cpp)
Example launch command with GPU acceleration and speculative drafting:
```bash
llama-server \
  -m ./models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -md ./models/mtp-gemma-4-12B-it.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngl 999 \
  -fa on \
  -fit off \
  --no-kv-unified \
  --port 9000 \
  -c 65536 \
  -np 2
```

### Option B: Ollama
```bash
# 1. Pull target model
ollama pull gemma:2b

# 2. Start Ollama server (listens on http://localhost:11434/v1)
ollama serve
```

### Option C: vLLM
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --port 8000
```

---

## ❓ FAQ & Troubleshooting

### `sqlite-vec` extension loading errors
If encountering issues loading `sqlite-vec` native extension on macOS:
1. Ensure Xcode Command Line Tools are installed:
   ```bash
   xcode-select --install
   ```
2. Reinstall `sqlite-vec` package:
   ```bash
   pip install --force-reinstall sqlite-vec
   ```
