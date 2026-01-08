"""Debug script to understand GPU memory usage."""
import os
import sys
import torch

# Force single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

def print_memory(stage):
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

print("="*60)
print("MEMORYLLM DEBUG - Memory Usage Analysis")
print("="*60)

print_memory("Start")

print("\n1. Loading model...")
from modeling_memoryllm import MemoryLLM
from transformers import AutoTokenizer

model_path = "YuWangX/memoryllm-8b"
model = MemoryLLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
print_memory("Model loaded")

print(f"\nModel config:")
print(f"  num_blocks: {model.num_blocks}")
print(f"  num_tokens: {model.num_tokens}")
print(f"  Total memory tokens: {model.num_blocks * model.num_tokens}")
print(f"  Memory pool shape: {model.memory.shape}")
print(f"  Memory pool size: {model.memory.numel() * 2 / 1e9:.2f} GB (bf16)")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
print_memory("Tokenizer loaded")

print("\n2. Testing single injection...")
test_text = "The quick brown fox jumps over the lazy dog. " * 20
context_ids = tokenizer(test_text, return_tensors="pt").input_ids.cuda()
print(f"   Context tokens: {context_ids.shape[1]}")

with torch.no_grad():
    model.inject_memory(context_ids, update_memory=True)
print_memory("After 1 injection")

print("\n3. Testing 5 more injections (simulating unrelated contexts)...")
for i in range(5):
    with torch.no_grad():
        model.inject_memory(context_ids, update_memory=True)
    print_memory(f"After injection {i+2}")

print("\n4. Testing generation...")
question = "What does the fox do?"
question_ids = tokenizer(question, return_tensors="pt").input_ids.cuda()
attention_mask = torch.cat([
    torch.ones(1, model.num_tokens * (model.num_blocks - 1)).cuda(),
    torch.ones(1, question_ids.shape[1]).cuda()
], dim=1)

with torch.no_grad():
    output = model.generate(
        inputs=question_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
print_memory("After generation")

print("\n5. Resetting memory...")
model.initialized.fill_(0)
torch.cuda.empty_cache()
print_memory("After reset + cache clear")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
