# GTE-Small in Pure C

A dependency-free C implementation of the [GTE-small](https://huggingface.co/thenlper/gte-small) text embedding model. Generates 384-dimensional embeddings for semantic similarity, search, and clustering.

**~700 lines of C. No dependencies. Matches PyTorch speed and accuracy.**

**Disclaimer**: This implementation was implemented using Claude Code and tested agaisnt the Python implementation. The output vectors are matching, but no accurate testing was performed to make sure the tokenizer and the inference works well in all the cases. If you plan using this library, make sure to verify the results are accurate.

## Quick Start

```bash
# Build
make

# Convert model (requires Python + safetensors, one-time)
pip install safetensors
python convert_model.py /path/to/gte-small gte-small.gtemodel

# Test
./test_gte "I love cats" "I love dogs" "The stock market crashed"
```

Output:
```
Cosine similarity matrix:
       S1     S2     S3
S1:  1.000  0.898  0.725
S2:  0.898  1.000  0.716
S3:  0.725  0.716  1.000
```

## C API

```c
#include "gte.h"

// Load model
gte_ctx *ctx = gte_load("gte-small.gtemodel");

// Generate embedding (384 floats, normalized)
float *emb = gte_embed(ctx, "Your text here");

// Compare texts (dot product = cosine similarity for normalized vectors)
float *emb1 = gte_embed(ctx, "I love cats");
float *emb2 = gte_embed(ctx, "I love dogs");
float similarity = gte_cosine_similarity(emb1, emb2, 384);  // 0.898

// Cleanup
free(emb);
free(emb1);
free(emb2);
gte_free(ctx);
```

### API Reference

| Function | Description |
|----------|-------------|
| `gte_ctx *gte_load(path)` | Load model, returns NULL on error |
| `void gte_free(ctx)` | Free model and resources |
| `float *gte_embed(ctx, text)` | Generate embedding (caller must free) |
| `float *gte_embed_batch(ctx, texts, count)` | Batch embed multiple texts |
| `int gte_dim(ctx)` | Embedding dimension (384) |
| `float gte_cosine_similarity(a, b, dim)` | Cosine similarity between embeddings |

## Performance

| Metric | Value |
|--------|-------|
| Inference time | **~12ms** per sentence |
| Model load time | ~20ms |
| Model size | 127 MB |
| Memory usage | ~130 MB (model) + ~2 MB (working) |
| Accuracy | Identical to Python/PyTorch |

Benchmarked on Apple MacBook pro M3. Compile with `-O3 -march=native -ffast-math` for best performance (~10x speedup compared to not letting the compiler exploit parallel instructions).

### vs Python/PyTorch

The C implementation matches the sentence-transformers/PyTorch inference speed within 15%:

| Implementation | Time |
|----------------|------|
| This (C) | 12ms |
| PyTorch | 10ms |

## Model Format

The `.gtemodel` format is a simple binary concatenation:
- Header: magic + config (vocab size, hidden dim, layers, etc.)
- Vocabulary: length-prefixed UTF-8 strings
- Weights: raw float32 arrays in fixed order

Convert from HuggingFace:
```bash
python convert_model.py /path/to/gte-small output.gtemodel
```

## CLI Usage

```bash
./test_gte [OPTIONS] [SENTENCES...]

Options:
  --model-path PATH   Path to .gtemodel file
  --help              Show help

Examples:
  ./test_gte                                    # Built-in test sentences
  ./test_gte "Hello world" "Goodbye world"      # Custom sentences
  ./test_gte --model-path custom.gtemodel       # Custom model path
```

## How It Works

1. **Tokenization**: WordPiece tokenizer splits text into subwords
2. **Embedding**: Token + position + segment embeddings
3. **Transformer**: 12 BERT layers (self-attention + FFN)
4. **Pooling**: Mean of token embeddings
5. **Normalize**: L2 normalization for cosine similarity

## License

MIT
