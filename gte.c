/*
 * GTE-Small Embedding Library Implementation
 *
 * Pure C implementation of BERT-based text embedding model.
 * No external dependencies.
 */

#include "gte.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define GTE_MAGIC "GTE1"
#define GTE_LAYER_NORM_EPS 1e-12f

/* Special token IDs */
#define TOKEN_PAD 0
#define TOKEN_UNK 100
#define TOKEN_CLS 101
#define TOKEN_SEP 102
#define TOKEN_MASK 103

/* Hash table size for vocabulary (prime number > vocab_size) */
#define VOCAB_HASH_SIZE 40009

/* ========================================================================
 * Data Structures
 * ======================================================================== */

/* Hash table entry for vocabulary lookup */
typedef struct {
    char *word;
    int id;
} vocab_entry;

/* Single transformer layer weights */
typedef struct {
    /* Self-attention */
    float *query_weight;      /* [hidden_size, hidden_size] */
    float *query_bias;        /* [hidden_size] */
    float *key_weight;        /* [hidden_size, hidden_size] */
    float *key_bias;          /* [hidden_size] */
    float *value_weight;      /* [hidden_size, hidden_size] */
    float *value_bias;        /* [hidden_size] */
    float *attn_output_weight; /* [hidden_size, hidden_size] */
    float *attn_output_bias;  /* [hidden_size] */
    float *attn_ln_weight;    /* [hidden_size] */
    float *attn_ln_bias;      /* [hidden_size] */

    /* Feed-forward network */
    float *ffn_inter_weight;  /* [intermediate_size, hidden_size] */
    float *ffn_inter_bias;    /* [intermediate_size] */
    float *ffn_output_weight; /* [hidden_size, intermediate_size] */
    float *ffn_output_bias;   /* [hidden_size] */
    float *ffn_ln_weight;     /* [hidden_size] */
    float *ffn_ln_bias;       /* [hidden_size] */
} layer_weights;

/* Main model context */
struct gte_ctx {
    /* Config */
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int intermediate_size;
    int max_seq_len;
    int head_dim;

    /* Vocabulary */
    char **vocab;             /* Array of vocabulary words */
    vocab_entry *vocab_hash;  /* Hash table for word -> id lookup */

    /* Embeddings */
    float *token_embeddings;  /* [vocab_size, hidden_size] */
    float *position_embeddings; /* [max_seq_len, hidden_size] */
    float *token_type_embeddings; /* [2, hidden_size] */
    float *embed_ln_weight;   /* [hidden_size] */
    float *embed_ln_bias;     /* [hidden_size] */

    /* Transformer layers */
    layer_weights *layers;

    /* Pooler (not used for embeddings but loaded) */
    float *pooler_weight;     /* [hidden_size, hidden_size] */
    float *pooler_bias;       /* [hidden_size] */

    /* Working memory for inference */
    float *hidden_states;     /* [max_seq_len, hidden_size] */
    float *attn_scores;       /* [num_heads, max_seq_len, max_seq_len] */
    float *q_proj;            /* [max_seq_len, hidden_size] */
    float *k_proj;            /* [max_seq_len, hidden_size] */
    float *v_proj;            /* [max_seq_len, hidden_size] */
    float *attn_output;       /* [max_seq_len, hidden_size] */
    float *ffn_hidden;        /* [max_seq_len, intermediate_size] */
    float *temp_hidden;       /* [max_seq_len, hidden_size] */
};

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/* FNV-1a hash for strings */
static uint32_t hash_string(const char *str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

/* Add word to vocabulary hash table */
static void vocab_hash_insert(vocab_entry *table, const char *word, int id) {
    uint32_t h = hash_string(word) % VOCAB_HASH_SIZE;
    while (table[h].word != NULL) {
        if (strcmp(table[h].word, word) == 0) {
            return; /* Already exists */
        }
        h = (h + 1) % VOCAB_HASH_SIZE;
    }
    table[h].word = strdup(word);
    table[h].id = id;
}

/* Look up word in vocabulary, returns -1 if not found */
static int vocab_lookup(vocab_entry *table, const char *word) {
    uint32_t h = hash_string(word) % VOCAB_HASH_SIZE;
    while (table[h].word != NULL) {
        if (strcmp(table[h].word, word) == 0) {
            return table[h].id;
        }
        h = (h + 1) % VOCAB_HASH_SIZE;
    }
    return -1;
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* Matrix-vector multiplication with bias: y = x @ W^T + b
 * x: [seq_len, in_dim], W: [out_dim, in_dim], b: [out_dim], y: [seq_len, out_dim]
 * Note: Weight matrices in BERT are stored as [out_dim, in_dim]
 */
static void linear(float *y, const float *x, const float *W, const float *b,
                   int seq_len, int in_dim, int out_dim) {
    for (int s = 0; s < seq_len; s++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = b ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x[s * in_dim + i] * W[o * in_dim + i];
            }
            y[s * out_dim + o] = sum;
        }
    }
}

/* Layer normalization */
static void layer_norm(float *out, const float *x, const float *gamma, const float *beta,
                       int seq_len, int hidden_size) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden_size;
        float *out_row = out + s * hidden_size;

        /* Compute mean */
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            mean += x_row[i];
        }
        mean /= hidden_size;

        /* Compute variance */
        float var = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = x_row[i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;

        /* Normalize and scale */
        float std_inv = 1.0f / sqrtf(var + GTE_LAYER_NORM_EPS);
        for (int i = 0; i < hidden_size; i++) {
            out_row[i] = gamma[i] * (x_row[i] - mean) * std_inv + beta[i];
        }
    }
}

/* GELU activation (approximate) */
static void gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        /* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
        x[i] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
    }
}

/* Softmax over last dimension */
static void softmax(float *x, int n) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

/* L2 normalize in place */
static void l2_normalize(float *x, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) {
        norm += x[i] * x[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < n; i++) {
            x[i] *= inv_norm;
        }
    }
}

/* ========================================================================
 * Tokenizer
 * ======================================================================== */

/* Check if character is punctuation (for basic tokenization) */
static int is_punctuation(unsigned char c) {
    return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
           (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
}

/* Check if character is whitespace */
static int is_whitespace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/* Basic tokenization: split on whitespace and punctuation, lowercase */
static char **basic_tokenize(const char *text, int *num_tokens) {
    int capacity = 64;
    char **tokens = malloc(capacity * sizeof(char *));
    int count = 0;

    const char *p = text;
    while (*p) {
        /* Skip whitespace */
        while (*p && is_whitespace(*p)) p++;
        if (!*p) break;

        /* Find end of token */
        const char *start = p;
        if (is_punctuation(*p)) {
            /* Single punctuation is a token */
            p++;
        } else {
            /* Read until whitespace or punctuation */
            while (*p && !is_whitespace(*p) && !is_punctuation(*p)) {
                p++;
            }
        }

        /* Create token (lowercase) */
        int len = p - start;
        char *token = malloc(len + 1);
        for (int i = 0; i < len; i++) {
            token[i] = tolower((unsigned char)start[i]);
        }
        token[len] = '\0';

        /* Add to list */
        if (count >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(char *));
        }
        tokens[count++] = token;
    }

    *num_tokens = count;
    return tokens;
}

/* WordPiece tokenization of a single word */
static int *wordpiece_tokenize(gte_ctx *ctx, const char *word, int *num_subtokens) {
    int len = strlen(word);
    if (len == 0) {
        *num_subtokens = 0;
        return NULL;
    }

    /* Buffer for subtokens (max = len subtokens) */
    int *subtokens = malloc((len + 1) * sizeof(int));
    int count = 0;

    int start = 0;
    while (start < len) {
        int end = len;
        int found_id = -1;

        /* Find longest matching subword */
        while (start < end) {
            /* Build candidate */
            char candidate[256];
            int cand_len = 0;

            if (start > 0) {
                candidate[cand_len++] = '#';
                candidate[cand_len++] = '#';
            }
            for (int i = start; i < end && cand_len < 254; i++) {
                candidate[cand_len++] = word[i];
            }
            candidate[cand_len] = '\0';

            int id = vocab_lookup(ctx->vocab_hash, candidate);
            if (id >= 0) {
                found_id = id;
                break;
            }
            end--;
        }

        if (found_id < 0) {
            /* No match found, use [UNK] for this character and move on */
            subtokens[count++] = TOKEN_UNK;
            start++;
        } else {
            subtokens[count++] = found_id;
            start = end;
        }
    }

    *num_subtokens = count;
    return subtokens;
}

/* Full tokenization: text -> token IDs */
static int *tokenize(gte_ctx *ctx, const char *text, int *num_tokens, int max_len) {
    /* Basic tokenization */
    int num_basic;
    char **basic_tokens = basic_tokenize(text, &num_basic);

    /* Collect all subtoken IDs */
    int capacity = 128;
    int *all_tokens = malloc(capacity * sizeof(int));
    int total = 0;

    /* Add [CLS] */
    all_tokens[total++] = TOKEN_CLS;

    /* WordPiece each token */
    for (int t = 0; t < num_basic && total < max_len - 1; t++) {
        int num_sub;
        int *subtokens = wordpiece_tokenize(ctx, basic_tokens[t], &num_sub);

        for (int s = 0; s < num_sub && total < max_len - 1; s++) {
            if (total >= capacity) {
                capacity *= 2;
                all_tokens = realloc(all_tokens, capacity * sizeof(int));
            }
            all_tokens[total++] = subtokens[s];
        }

        free(subtokens);
        free(basic_tokens[t]);
    }
    free(basic_tokens);

    /* Add [SEP] */
    if (total >= capacity) {
        capacity *= 2;
        all_tokens = realloc(all_tokens, capacity * sizeof(int));
    }
    all_tokens[total++] = TOKEN_SEP;

    *num_tokens = total;
    return all_tokens;
}

/* ========================================================================
 * Transformer Forward Pass
 * ======================================================================== */

/* Self-attention for a single layer */
static void self_attention(gte_ctx *ctx, layer_weights *layer, int seq_len, const int *attn_mask) {
    int hidden = ctx->hidden_size;
    int heads = ctx->num_heads;
    int head_dim = ctx->head_dim;

    /* Project Q, K, V */
    linear(ctx->q_proj, ctx->hidden_states, layer->query_weight, layer->query_bias,
           seq_len, hidden, hidden);
    linear(ctx->k_proj, ctx->hidden_states, layer->key_weight, layer->key_bias,
           seq_len, hidden, hidden);
    linear(ctx->v_proj, ctx->hidden_states, layer->value_weight, layer->value_bias,
           seq_len, hidden, hidden);

    /* Compute attention for each head */
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < heads; h++) {
        /* Attention scores for this head: Q @ K^T / sqrt(d_k) */
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    int q_idx = i * hidden + h * head_dim + d;
                    int k_idx = j * hidden + h * head_dim + d;
                    score += ctx->q_proj[q_idx] * ctx->k_proj[k_idx];
                }
                score *= scale;

                /* Apply attention mask */
                if (attn_mask && !attn_mask[j]) {
                    score = -10000.0f;
                }

                ctx->attn_scores[h * seq_len * seq_len + i * seq_len + j] = score;
            }

            /* Softmax over keys */
            softmax(&ctx->attn_scores[h * seq_len * seq_len + i * seq_len], seq_len);
        }

        /* Weighted sum of values */
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float attn = ctx->attn_scores[h * seq_len * seq_len + i * seq_len + j];
                    int v_idx = j * hidden + h * head_dim + d;
                    sum += attn * ctx->v_proj[v_idx];
                }
                ctx->attn_output[i * hidden + h * head_dim + d] = sum;
            }
        }
    }

    /* Output projection */
    linear(ctx->temp_hidden, ctx->attn_output, layer->attn_output_weight, layer->attn_output_bias,
           seq_len, hidden, hidden);

    /* Residual connection and layer norm */
    for (int i = 0; i < seq_len * hidden; i++) {
        ctx->temp_hidden[i] += ctx->hidden_states[i];
    }
    layer_norm(ctx->hidden_states, ctx->temp_hidden, layer->attn_ln_weight, layer->attn_ln_bias,
               seq_len, hidden);
}

/* Feed-forward network for a single layer */
static void feed_forward(gte_ctx *ctx, layer_weights *layer, int seq_len) {
    int hidden = ctx->hidden_size;
    int inter = ctx->intermediate_size;

    /* Intermediate layer */
    linear(ctx->ffn_hidden, ctx->hidden_states, layer->ffn_inter_weight, layer->ffn_inter_bias,
           seq_len, hidden, inter);
    gelu(ctx->ffn_hidden, seq_len * inter);

    /* Output layer */
    linear(ctx->temp_hidden, ctx->ffn_hidden, layer->ffn_output_weight, layer->ffn_output_bias,
           seq_len, inter, hidden);

    /* Residual connection and layer norm */
    for (int i = 0; i < seq_len * hidden; i++) {
        ctx->temp_hidden[i] += ctx->hidden_states[i];
    }
    layer_norm(ctx->hidden_states, ctx->temp_hidden, layer->ffn_ln_weight, layer->ffn_ln_bias,
               seq_len, hidden);
}

/* Full transformer forward pass */
static void transformer_forward(gte_ctx *ctx, const int *token_ids, int seq_len, const int *attn_mask) {
    int hidden = ctx->hidden_size;

    /* Compute embeddings */
    for (int s = 0; s < seq_len; s++) {
        int token_id = token_ids[s];
        for (int d = 0; d < hidden; d++) {
            ctx->hidden_states[s * hidden + d] =
                ctx->token_embeddings[token_id * hidden + d] +
                ctx->position_embeddings[s * hidden + d] +
                ctx->token_type_embeddings[d]; /* token_type = 0 */
        }
    }

    /* Embedding layer norm */
    layer_norm(ctx->hidden_states, ctx->hidden_states, ctx->embed_ln_weight, ctx->embed_ln_bias,
               seq_len, hidden);

    /* Process each transformer layer */
    for (int l = 0; l < ctx->num_layers; l++) {
        self_attention(ctx, &ctx->layers[l], seq_len, attn_mask);
        feed_forward(ctx, &ctx->layers[l], seq_len);
    }
}

/* Mean pooling */
static void mean_pooling(float *output, const float *hidden_states, const int *attn_mask,
                         int seq_len, int hidden_size) {
    /* Initialize output to zero */
    memset(output, 0, hidden_size * sizeof(float));

    /* Sum up hidden states for non-padded tokens */
    int count = 0;
    for (int s = 0; s < seq_len; s++) {
        if (attn_mask[s]) {
            for (int d = 0; d < hidden_size; d++) {
                output[d] += hidden_states[s * hidden_size + d];
            }
            count++;
        }
    }

    /* Average */
    if (count > 0) {
        float inv_count = 1.0f / count;
        for (int d = 0; d < hidden_size; d++) {
            output[d] *= inv_count;
        }
    }
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

static int read_uint32(FILE *f, int *val) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    *val = buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24);
    return 1;
}

static int read_uint16(FILE *f, int *val) {
    unsigned char buf[2];
    if (fread(buf, 1, 2, f) != 2) return 0;
    *val = buf[0] | (buf[1] << 8);
    return 1;
}

static float *read_floats(FILE *f, int count) {
    float *data = malloc(count * sizeof(float));
    if (!data) return NULL;
    if (fread(data, sizeof(float), count, f) != (size_t)count) {
        free(data);
        return NULL;
    }
    return data;
}

gte_ctx *gte_load(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "gte_load: cannot open %s\n", model_path);
        return NULL;
    }

    /* Check magic */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, GTE_MAGIC, 4) != 0) {
        fprintf(stderr, "gte_load: invalid magic\n");
        fclose(f);
        return NULL;
    }

    /* Allocate context */
    gte_ctx *ctx = calloc(1, sizeof(gte_ctx));
    if (!ctx) {
        fclose(f);
        return NULL;
    }

    /* Read config */
    if (!read_uint32(f, &ctx->vocab_size) ||
        !read_uint32(f, &ctx->hidden_size) ||
        !read_uint32(f, &ctx->num_layers) ||
        !read_uint32(f, &ctx->num_heads) ||
        !read_uint32(f, &ctx->intermediate_size) ||
        !read_uint32(f, &ctx->max_seq_len)) {
        fprintf(stderr, "gte_load: failed to read config\n");
        goto error;
    }

    ctx->head_dim = ctx->hidden_size / ctx->num_heads;

    /* Read vocabulary */
    ctx->vocab = malloc(ctx->vocab_size * sizeof(char *));
    ctx->vocab_hash = calloc(VOCAB_HASH_SIZE, sizeof(vocab_entry));
    if (!ctx->vocab || !ctx->vocab_hash) goto error;

    for (int i = 0; i < ctx->vocab_size; i++) {
        int len;
        if (!read_uint16(f, &len)) goto error;

        ctx->vocab[i] = malloc(len + 1);
        if (!ctx->vocab[i]) goto error;
        if (fread(ctx->vocab[i], 1, len, f) != (size_t)len) goto error;
        ctx->vocab[i][len] = '\0';

        vocab_hash_insert(ctx->vocab_hash, ctx->vocab[i], i);
    }

    /* Read embeddings */
    ctx->token_embeddings = read_floats(f, ctx->vocab_size * ctx->hidden_size);
    ctx->position_embeddings = read_floats(f, ctx->max_seq_len * ctx->hidden_size);
    ctx->token_type_embeddings = read_floats(f, 2 * ctx->hidden_size);
    ctx->embed_ln_weight = read_floats(f, ctx->hidden_size);
    ctx->embed_ln_bias = read_floats(f, ctx->hidden_size);

    if (!ctx->token_embeddings || !ctx->position_embeddings ||
        !ctx->token_type_embeddings || !ctx->embed_ln_weight || !ctx->embed_ln_bias) {
        goto error;
    }

    /* Read transformer layers */
    ctx->layers = malloc(ctx->num_layers * sizeof(layer_weights));
    if (!ctx->layers) goto error;
    memset(ctx->layers, 0, ctx->num_layers * sizeof(layer_weights));

    for (int l = 0; l < ctx->num_layers; l++) {
        layer_weights *layer = &ctx->layers[l];

        layer->query_weight = read_floats(f, ctx->hidden_size * ctx->hidden_size);
        layer->query_bias = read_floats(f, ctx->hidden_size);
        layer->key_weight = read_floats(f, ctx->hidden_size * ctx->hidden_size);
        layer->key_bias = read_floats(f, ctx->hidden_size);
        layer->value_weight = read_floats(f, ctx->hidden_size * ctx->hidden_size);
        layer->value_bias = read_floats(f, ctx->hidden_size);
        layer->attn_output_weight = read_floats(f, ctx->hidden_size * ctx->hidden_size);
        layer->attn_output_bias = read_floats(f, ctx->hidden_size);
        layer->attn_ln_weight = read_floats(f, ctx->hidden_size);
        layer->attn_ln_bias = read_floats(f, ctx->hidden_size);

        layer->ffn_inter_weight = read_floats(f, ctx->intermediate_size * ctx->hidden_size);
        layer->ffn_inter_bias = read_floats(f, ctx->intermediate_size);
        layer->ffn_output_weight = read_floats(f, ctx->hidden_size * ctx->intermediate_size);
        layer->ffn_output_bias = read_floats(f, ctx->hidden_size);
        layer->ffn_ln_weight = read_floats(f, ctx->hidden_size);
        layer->ffn_ln_bias = read_floats(f, ctx->hidden_size);

        if (!layer->query_weight || !layer->query_bias ||
            !layer->key_weight || !layer->key_bias ||
            !layer->value_weight || !layer->value_bias ||
            !layer->attn_output_weight || !layer->attn_output_bias ||
            !layer->attn_ln_weight || !layer->attn_ln_bias ||
            !layer->ffn_inter_weight || !layer->ffn_inter_bias ||
            !layer->ffn_output_weight || !layer->ffn_output_bias ||
            !layer->ffn_ln_weight || !layer->ffn_ln_bias) {
            goto error;
        }
    }

    /* Read pooler (not used for embeddings) */
    ctx->pooler_weight = read_floats(f, ctx->hidden_size * ctx->hidden_size);
    ctx->pooler_bias = read_floats(f, ctx->hidden_size);

    fclose(f);

    /* Allocate working memory */
    int max_seq = ctx->max_seq_len;
    int hidden = ctx->hidden_size;
    int inter = ctx->intermediate_size;
    int heads = ctx->num_heads;

    ctx->hidden_states = malloc(max_seq * hidden * sizeof(float));
    ctx->attn_scores = malloc(heads * max_seq * max_seq * sizeof(float));
    ctx->q_proj = malloc(max_seq * hidden * sizeof(float));
    ctx->k_proj = malloc(max_seq * hidden * sizeof(float));
    ctx->v_proj = malloc(max_seq * hidden * sizeof(float));
    ctx->attn_output = malloc(max_seq * hidden * sizeof(float));
    ctx->ffn_hidden = malloc(max_seq * inter * sizeof(float));
    ctx->temp_hidden = malloc(max_seq * hidden * sizeof(float));

    if (!ctx->hidden_states || !ctx->attn_scores || !ctx->q_proj ||
        !ctx->k_proj || !ctx->v_proj || !ctx->attn_output ||
        !ctx->ffn_hidden || !ctx->temp_hidden) {
        gte_free(ctx);
        return NULL;
    }

    return ctx;

error:
    fprintf(stderr, "gte_load: error reading model\n");
    fclose(f);
    gte_free(ctx);
    return NULL;
}

void gte_free(gte_ctx *ctx) {
    if (!ctx) return;

    /* Free vocabulary */
    if (ctx->vocab) {
        for (int i = 0; i < ctx->vocab_size; i++) {
            free(ctx->vocab[i]);
        }
        free(ctx->vocab);
    }
    if (ctx->vocab_hash) {
        for (int i = 0; i < VOCAB_HASH_SIZE; i++) {
            free(ctx->vocab_hash[i].word);
        }
        free(ctx->vocab_hash);
    }

    /* Free embeddings */
    free(ctx->token_embeddings);
    free(ctx->position_embeddings);
    free(ctx->token_type_embeddings);
    free(ctx->embed_ln_weight);
    free(ctx->embed_ln_bias);

    /* Free layers */
    if (ctx->layers) {
        for (int l = 0; l < ctx->num_layers; l++) {
            layer_weights *layer = &ctx->layers[l];
            free(layer->query_weight);
            free(layer->query_bias);
            free(layer->key_weight);
            free(layer->key_bias);
            free(layer->value_weight);
            free(layer->value_bias);
            free(layer->attn_output_weight);
            free(layer->attn_output_bias);
            free(layer->attn_ln_weight);
            free(layer->attn_ln_bias);
            free(layer->ffn_inter_weight);
            free(layer->ffn_inter_bias);
            free(layer->ffn_output_weight);
            free(layer->ffn_output_bias);
            free(layer->ffn_ln_weight);
            free(layer->ffn_ln_bias);
        }
        free(ctx->layers);
    }

    /* Free pooler */
    free(ctx->pooler_weight);
    free(ctx->pooler_bias);

    /* Free working memory */
    free(ctx->hidden_states);
    free(ctx->attn_scores);
    free(ctx->q_proj);
    free(ctx->k_proj);
    free(ctx->v_proj);
    free(ctx->attn_output);
    free(ctx->ffn_hidden);
    free(ctx->temp_hidden);

    free(ctx);
}

/* ========================================================================
 * Public API
 * ======================================================================== */

float *gte_embed(gte_ctx *ctx, const char *text) {
    if (!ctx || !text) return NULL;

    /* Tokenize */
    int num_tokens;
    int *token_ids = tokenize(ctx, text, &num_tokens, ctx->max_seq_len);
    if (!token_ids) return NULL;

    /* Create attention mask */
    int *attn_mask = malloc(num_tokens * sizeof(int));
    for (int i = 0; i < num_tokens; i++) {
        attn_mask[i] = 1;
    }

    /* Run transformer */
    transformer_forward(ctx, token_ids, num_tokens, attn_mask);

    /* Mean pooling */
    float *embedding = malloc(ctx->hidden_size * sizeof(float));
    if (!embedding) {
        free(token_ids);
        free(attn_mask);
        return NULL;
    }
    mean_pooling(embedding, ctx->hidden_states, attn_mask, num_tokens, ctx->hidden_size);

    /* L2 normalize */
    l2_normalize(embedding, ctx->hidden_size);

    free(token_ids);
    free(attn_mask);

    return embedding;
}

float *gte_embed_batch(gte_ctx *ctx, const char **texts, int count) {
    if (!ctx || !texts || count <= 0) return NULL;

    float *embeddings = malloc(count * ctx->hidden_size * sizeof(float));
    if (!embeddings) return NULL;

    for (int i = 0; i < count; i++) {
        float *emb = gte_embed(ctx, texts[i]);
        if (!emb) {
            free(embeddings);
            return NULL;
        }
        memcpy(embeddings + i * ctx->hidden_size, emb, ctx->hidden_size * sizeof(float));
        free(emb);
    }

    return embeddings;
}

int gte_dim(gte_ctx *ctx) {
    return ctx ? ctx->hidden_size : 0;
}

int gte_max_seq_len(gte_ctx *ctx) {
    return ctx ? ctx->max_seq_len : 0;
}

float gte_cosine_similarity(const float *a, const float *b, int dim) {
    /* Assumes normalized vectors, so dot product = cosine similarity */
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}
