# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

`verifiers` is a framework for training LLMs with reinforcement learning in verifiable multi-turn environments via Group-Relative Policy Optimization (GRPO). The implementation builds upon the base `transformers` Trainer and is optimized for efficient async multi-turn inference and training with off-policy overlapping.

## Common Development Commands

### Installation
```bash
# Full installation with all dependencies
uv sync --extra all && uv pip install flash-attn --no-build-isolation

# CPU-only development (API-only, no training)
uv add verifiers
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m unit        # Run unit tests only
pytest -m integration # Run integration tests only
```

### Building Documentation
```bash
cd docs
make html  # Build HTML documentation
```

### Package Building
```bash
# Build the package
python -m build

# Upload to PyPI
twine upload dist/*
```

### Training Workflow
```bash
# Step 1: Launch vLLM inference server (using GPUs 0,1)
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct' --tensor-parallel-size 2

# Step 2: Launch training script (using GPUs 2,3)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml train.py
```

## Code Architecture

### Core Components

1. **Environments** (`verifiers/envs/`)
   - `Environment`: Abstract base class for all environments
   - `SingleTurnEnv`: Single-turn reasoning tasks
   - `MultiTurnEnv`: Multi-turn conversation environments
   - `ToolEnv`: Multi-turn tool use with custom Python functions
   - `SmolaToolEnv`: Integration with Hugging Face smolagents
   - `CodeMathEnv`: Interactive Python execution environment
   - `ReasoningGymEnv`: Direct training for reasoning-gym tasks

2. **Trainers** (`verifiers/trainers/`)
   - `GRPOTrainer`: Main GRPO trainer built on transformers.Trainer
   - `GRPOConfig`: Configuration for GRPO training
   - `AsyncBatchGenerator`: Async batch generation for efficient training
   - `AsyncDataLoaderWrapper`: Wrapper for async data loading

3. **Parsers** (`verifiers/parsers/`)
   - `Parser`: Base parser class
   - `XMLParser`: Parses XML-formatted responses (e.g., `<think>...</think>`)
   - `ThinkParser`: Specialized parser for reasoning traces
   - `SmolaParser`: Parser for smolagents format

4. **Rubrics** (`verifiers/rubrics/`)
   - `Rubric`: Base class for reward functions
   - `MathRubric`: Math-specific reward functions
   - `JudgeRubric`: LLM-as-judge reward functions
   - `ToolRubric`: Tool use specific rewards
   - `RubricGroup`: Combines multiple rubrics

5. **Tools** (`verifiers/tools/`)
   - `python`: Python code execution tool
   - `search`: Web search capabilities
   - `calculator`: Mathematical calculations
   - `ask`: Q&A interaction tool

6. **Inference** (`verifiers/inference/`)
   - `vllm_server`: vLLM server implementation (launched via `vf-vllm` command)
   - `vllm_client`: Client for connecting to vLLM servers

### Key Design Patterns

1. **Environment-Rubric-Parser Pattern**: Environments encapsulate tasks, parsers handle response formatting, and rubrics compute rewards.

2. **Async-First Design**: All rollouts and reward calculations support async execution for efficiency.

3. **Modular Tool System**: Tools are Python functions that can be easily added to any ToolEnv.

4. **OpenAI-Compatible API**: All inference servers and clients use OpenAI-compatible endpoints.

## Environment Variables

- `OPENAI_API_KEY`: Required for vLLM server (can be dummy value)
- `WANDB_API_KEY`: For wandb logging (optional)
- `HF_TOKEN`: For accessing gated models
- `NCCL_P2P_DISABLE`: Set to 1 if experiencing NCCL communication issues

## Training Best Practices

1. **Memory Management**: Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` to lower memory usage.

2. **Hyperparameter Guidelines**:
   - Start with `beta=0.001` for KL penalty
   - Use `num_generations>=4` for diversity
   - Set `learning_rate=1e-6` initially
   - Enable `sync_ref_model=True` for long runs

3. **Debugging**:
   - Set `log_completions=True` to see generated samples
   - Use `report_to=None` to disable wandb during testing
   - Check reward diversity with `scale_rewards=True`

## Common Troubleshooting

1. **NCCL Issues**: 
   ```bash
   export NCCL_P2P_DISABLE=1
   ```

2. **Memory Errors**: Reduce batch size or use gradient accumulation

3. **vLLM Connection**: Ensure server is running before training script

4. **Timeout Errors**: Increase `async_generation_timeout` in config

5. **Single GPU Setup**: For single GPU inference server:
   ```bash
   CUDA_VISIBLE_DEVICES=0 vf-vllm --model 'Qwen/Qwen2.5-1.5B-Instruct' --tensor-parallel-size 1
   ```

6. **Flash Attention Issues**: If you encounter flash-attn installation problems:
   ```bash
   uv sync --extra all && uv pip install flash-attn==2.7.2.post1 --no-build-isolation
   ```

7. **Python 3.12 Installation** (Ubuntu/Debian):
   ```bash
   sudo apt update
   sudo apt install python3.12 python3.12-venv python3.12-dev
   ```

8. **DeepSpeed Build Issues**: If DeepSpeed fails to build:
   ```bash
   DS_BUILD_OPS=0 DS_BUILD_CPU_ADAM=1 uv pip install deepspeed --no-cache
   ```