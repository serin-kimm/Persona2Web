# AgentOccam Installation Guide

## 🚀 Complete Installation for All Config Modes

This guide will help you install all dependencies needed to run:
- `on-demand_o3_1.yml` (On-demand memory mode)
- `pre-execution_o3_1.yml` (Pre-execution memory mode)  
- `no_history_o3_1.yml` (No memory history mode)

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning repositories)

## 🔧 Installation Steps

### 1. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Install Playwright Browsers

```bash
# Install Playwright browser binaries
playwright install
```

### 3. Environment Setup

Create a `.env` file or set environment variables:

```bash
# OpenAI API Key (required for GPT models)
export OPENAI_API_KEY="your_openai_api_key_here"

# Optional: Other LLM API Keys
export ANTHROPIC_API_KEY="your_claude_api_key_here"
export GOOGLE_API_KEY="your_google_ai_api_key_here"
export COHERE_API_KEY="your_cohere_api_key_here"

# Optional: AWS credentials for Claude via Bedrock
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 4. Setup Memory Environment (for memory modes)

```bash
# Setup embedding environment for memory retrieval
python setup_embedding_env.py
```

### 5. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## 🎯 Running the Different Modes

### On-demand Mode
```bash
python agent_runner.py --config AgentOccam/configs/on-demand_gpt4.1_1.yml
```

### Pre-execution Mode
```bash
python agent_runner.py --config AgentOccam/configs/pre-execution_gpt4.1_1.yml
```

### No History Mode
```bash
python agent_runner.py --config AgentOccam/configs/no_history_gpt4.1_1.yml
```

## 🔍 Troubleshooting

### Common Issues

1. **Playwright browser not found**:
   ```bash
   playwright install chromium
   ```

2. **FAISS installation issues**:
   ```bash
   # For CPU-only version
   pip install faiss-cpu
   
   # For GPU version (if you have CUDA)
   pip install faiss-gpu
   ```

3. **Torch installation issues**:
   ```bash
   # For CPU-only version
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA version
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Memory retriever not available**:
   - Ensure `OPENAI_API_KEY` is set
   - Run `python setup_embedding_env.py` to initialize embedding environment
   - Check that `sentence-transformers` and `faiss-cpu` are properly installed

## 📦 Package Overview

### Core Dependencies
- **Web Automation**: playwright, playwright-stealth
- **AI/ML**: torch, transformers, sentence-transformers, faiss-cpu
- **LLM APIs**: openai, anthropic, google-generativeai, cohere

### Optional Dependencies
- **Advanced ML**: ctranslate2, bitsandbytes, peft (commented out by default)
- **Development**: pytest, black, flake8 (commented out by default)

## 🎛️ Configuration

Each config file supports different memory modes:
- **on-demand**: Memory retrieval during task execution
- **pre-execution**: Memory retrieval before task execution  
- **no-history**: No memory functionality (fastest)

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install browsers
playwright install

# 3. Set API key
export OPENAI_API_KEY="your_key_here"

# 4. Setup memory environment
python setup_embedding_env.py

# 5. Run AgentOccam
python agent_runner.py --config AgentOccam/configs/on-demand_gpt4.1_1.yml
```

## 🆘 Support

If you encounter issues:
1. Check that all environment variables are set
2. Verify browser installation: `playwright install --help`
3. Test individual components: `python -c "import torch, sentence_transformers, openai; print('All imports successful')"`
