#!/usr/bin/env python3
"""
Convert GTE-small model from safetensors to .gtemodel binary format.

Usage:
    python convert_model.py offline/local_complete_model gte-small.gtemodel
"""

import sys
import struct
import json
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Please install safetensors: pip install safetensors")
    sys.exit(1)

def load_vocab(vocab_path):
    """Load vocabulary from vocab.txt"""
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.rstrip('\n'))
    return vocab

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.gtemodel>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    output_path = sys.argv[2]

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]  # 30522
    hidden_size = config["hidden_size"]  # 384
    num_layers = config["num_hidden_layers"]  # 12
    num_heads = config["num_attention_heads"]  # 12
    intermediate_size = config["intermediate_size"]  # 1536
    max_seq_length = config["max_position_embeddings"]  # 512

    print(f"Model config:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  max_seq_length: {max_seq_length}")

    # Load vocabulary
    vocab = load_vocab(model_dir / "vocab.txt")
    assert len(vocab) == vocab_size, f"Vocab size mismatch: {len(vocab)} vs {vocab_size}"
    print(f"  Loaded {len(vocab)} vocabulary entries")

    # Load safetensors
    safetensors_path = model_dir / "model.safetensors"
    tensors = safe_open(safetensors_path, framework="numpy")

    # Print all tensor names for inspection
    print("\nTensor names in safetensors:")
    for name in sorted(tensors.keys()):
        shape = tensors.get_tensor(name).shape
        print(f"  {name}: {shape}")

    # Write binary format
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GTE1')  # Magic
        f.write(struct.pack('<I', vocab_size))
        f.write(struct.pack('<I', hidden_size))
        f.write(struct.pack('<I', num_layers))
        f.write(struct.pack('<I', num_heads))
        f.write(struct.pack('<I', intermediate_size))
        f.write(struct.pack('<I', max_seq_length))

        # Vocabulary
        for word in vocab:
            word_bytes = word.encode('utf-8')
            f.write(struct.pack('<H', len(word_bytes)))
            f.write(word_bytes)

        def write_tensor(name):
            tensor = tensors.get_tensor(name).astype('float32')
            f.write(tensor.tobytes())
            return tensor.shape

        # Embeddings
        print("\nWriting embeddings...")
        write_tensor("embeddings.word_embeddings.weight")  # [30522, 384]
        write_tensor("embeddings.position_embeddings.weight")  # [512, 384]
        write_tensor("embeddings.token_type_embeddings.weight")  # [2, 384]
        write_tensor("embeddings.LayerNorm.weight")  # [384]
        write_tensor("embeddings.LayerNorm.bias")  # [384]

        # Transformer layers
        print("Writing transformer layers...")
        for layer_idx in range(num_layers):
            prefix = f"encoder.layer.{layer_idx}"

            # Attention
            write_tensor(f"{prefix}.attention.self.query.weight")  # [384, 384]
            write_tensor(f"{prefix}.attention.self.query.bias")  # [384]
            write_tensor(f"{prefix}.attention.self.key.weight")  # [384, 384]
            write_tensor(f"{prefix}.attention.self.key.bias")  # [384]
            write_tensor(f"{prefix}.attention.self.value.weight")  # [384, 384]
            write_tensor(f"{prefix}.attention.self.value.bias")  # [384]
            write_tensor(f"{prefix}.attention.output.dense.weight")  # [384, 384]
            write_tensor(f"{prefix}.attention.output.dense.bias")  # [384]
            write_tensor(f"{prefix}.attention.output.LayerNorm.weight")  # [384]
            write_tensor(f"{prefix}.attention.output.LayerNorm.bias")  # [384]

            # FFN
            write_tensor(f"{prefix}.intermediate.dense.weight")  # [1536, 384]
            write_tensor(f"{prefix}.intermediate.dense.bias")  # [1536]
            write_tensor(f"{prefix}.output.dense.weight")  # [384, 1536]
            write_tensor(f"{prefix}.output.dense.bias")  # [384]
            write_tensor(f"{prefix}.output.LayerNorm.weight")  # [384]
            write_tensor(f"{prefix}.output.LayerNorm.bias")  # [384]

            print(f"  Layer {layer_idx} done")

        # Pooler (not used for embeddings but included for completeness)
        write_tensor("pooler.dense.weight")  # [384, 384]
        write_tensor("pooler.dense.bias")  # [384]

    print(f"\nModel saved to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
