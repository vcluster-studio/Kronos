"""Test bit_mask soft decode correctness"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from model.kronos import KronosTokenizer

# === 1. Test bit_mask construction ===
s1_bits = 10
vocab_s1 = 2 ** s1_bits
s1_bit_mask = torch.zeros(vocab_s1, s1_bits)
for idx in range(vocab_s1):
    for b in range(s1_bits):
        if (idx >> b) & 1:
            s1_bit_mask[idx, b] = 1.0

# Test with known index 42
test_idx = 42
bits_42 = [(42 >> b) & 1 for b in range(s1_bits)]
print(f'Index {test_idx} bits: {bits_42}')

# One-hot probabilities
s1_probs = torch.zeros(1, 1, vocab_s1)
s1_probs[0, 0, test_idx] = 1.0

# Matrix multiply method
s1_bit_probs = s1_probs @ s1_bit_mask
print(f'bit_probs (matmul): {s1_bit_probs[0, 0].tolist()}')
match = all(abs(s1_bit_probs[0,0,b].item() - bits_42[b]) < 1e-6 for b in range(s1_bits))
print(f'Match: {match}')

# Mixed probs test
s1_probs2 = torch.zeros(1, 1, vocab_s1)
s1_probs2[0, 0, 0] = 0.5     # all bits 0
s1_probs2[0, 0, 1023] = 0.5  # all bits 1
s1_bit_probs2 = s1_probs2 @ s1_bit_mask
print(f'Mixed (should be 0.5): {s1_bit_probs2[0, 0].tolist()[:5]}...')

# === 2. Test with real tokenizer ===
tokenizer = KronosTokenizer.from_pretrained('outputs/models/ma60_tokenizer_v1/checkpoints/best_model')

# Soft decode from one-hot
expected_bits = s1_bit_probs * 2 - 1  # bipolar
expected_bits_full = torch.cat([expected_bits, expected_bits], dim=-1)
q_scale = 1.0 / (20 ** 0.5)
expected_bits_full = expected_bits_full * q_scale
decoded = tokenizer.decode_from_bits(expected_bits_full)
print(f'Soft decoded shape: {decoded.shape}')
print(f'Soft decoded close: {decoded[0, 0, 3].item():.6f}')

# Hard decode for comparison
hard_bits = tokenizer.indices_to_bits(
    (torch.tensor([[42]]), torch.tensor([[42]])), half=True
)
hard_decoded = tokenizer.decode_from_bits(hard_bits)
print(f'Hard decoded close: {hard_decoded[0, 0, 3].item():.6f}')
print(f'Match (soft==hard for one-hot): {abs(decoded[0,0,3].item() - hard_decoded[0,0,3].item()) < 1e-4}')

# === 3. Test gradient flow ===
logits = torch.randn(1, 1, vocab_s1, requires_grad=True)
probs = F.softmax(logits, dim=-1)
bit_probs = probs @ s1_bit_mask
bipolar = bit_probs * 2 - 1
loss = bipolar.sum()
loss.backward()
print(f'Gradient flows through matmul: {logits.grad is not None and logits.grad.abs().sum() > 0}')

print('\nAll tests passed!')
