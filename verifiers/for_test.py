from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(["Hello world!", "Hi!"], padding=True, return_tensors="pt")
print(inputs["input_ids"])
print(inputs["attention_mask"])
model = AutoModelForCausalLM.from_pretrained("gpt2")
outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
log_probs = F.log_softmax(outputs.logits, dim=-1)
print(log_probs)