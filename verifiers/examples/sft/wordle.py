import verifiers as vf
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import DownloadMode
from datasets import config as datasets_config

"""
accelerate launch --config-file configs/zero3.yaml --num-processes 8 verifiers/examples/sft/wordle.py
"""

# convenience function for FA2 initialization
# model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen3-1.7B", use_liger=False)
# model = model.to('cuda')  # Move model to GPU for Flash Attention 2.0
tokenizer = vf.get_tokenizer("Qwen/Qwen3-1.7B")
dataset = load_dataset('willcb/V3-wordle', split='train[:10]', cache_dir=datasets_config.HF_DATASETS_CACHE, download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS)

def generate_conversation(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]
    conversations = []
    for prompt, completion in zip(prompts, completions):
        # Create a conversation with the prompt and completion
        conversations.append(
            prompt + completion
        )
    return { "conversations": conversations}

reasoning_conversations = tokenizer.apply_chat_template(
    dataset.map(generate_conversation, batched = True)["conversations"],
    tokenize=False,
    enable_thinking=False
)

print(reasoning_conversations[0])
import sys
sys.exit()

tok_counts = []
for row in dataset:
    # count tokens in (prompt, completion)
    messages = row['prompt'] + row['completion'] # type: ignore
    toks = tokenizer.apply_chat_template( 
        messages,
        enable_thinking=False,
        tokenize=True
    )
    tok_counts.append(len(toks))

# def formatting_func_with_thinking(example):
#     """Apply chat template with enable_thinking=False"""
#     messages = example["prompt"] + example["completion"]
#     return tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False,
#         add_generation_prompt=False,
#         enable_thinking=False
#     )

# tok count stats
print(f"Dataset size: {len(tok_counts)}")
print(f"Min tokens: {min(tok_counts)}")
print(f"Max tokens: {max(tok_counts)}")
print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

args = SFTConfig(
    max_length=8192,
    output_dir="outputs/model/sft-wordle",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    max_grad_norm=0.1,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=1,
    save_only_model=True,
    log_on_each_node=True,
    completion_only_loss=True
    )


# Create data collator that masks everything except assistant responses
# collator = DataCollatorForCompletionOnlyLM(
#     instruction_template="<|im_start|>user",
#     response_template="<|im_start|>assistant", 
#     tokenizer=tokenizer,
#     mlm=False
# )

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=dataset # type: ignore
)
trainer.train()