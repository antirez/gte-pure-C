/*
 * Test program for GTE library
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gte.h"

#define DEFAULT_MODEL_PATH "gte-small.gtemodel"
#define MAX_SENTENCES 64

void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS] [SENTENCES...]\n\n", prog);
    printf("Test GTE-small embedding model by computing embeddings and similarity matrix.\n\n");
    printf("Options:\n");
    printf("  --model-path PATH   Path to .gtemodel file (default: %s)\n", DEFAULT_MODEL_PATH);
    printf("  --help              Show this help message\n\n");
    printf("Arguments:\n");
    printf("  SENTENCES           One or more sentences to embed (quote each sentence)\n");
    printf("                      If none provided, uses built-in example sentences\n\n");
    printf("Examples:\n");
    printf("  %s\n", prog);
    printf("  %s \"Hello world\" \"Goodbye world\"\n", prog);
    printf("  %s --model-path my-model.gtemodel \"Test sentence\"\n", prog);
}

void print_embedding(const float *emb, int dim, int n) {
    printf("[");
    for (int i = 0; i < n && i < dim; i++) {
        printf("%.6f", emb[i]);
        if (i < n - 1) printf(", ");
    }
    if (n < dim) printf(", ...");
    printf("]\n");
}

int main(int argc, char **argv) {
    const char *model_path = DEFAULT_MODEL_PATH;
    const char *user_sentences[MAX_SENTENCES];
    int num_user_sentences = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model-path") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --model-path requires an argument\n");
                return 1;
            }
            model_path = argv[++i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
            return 1;
        } else {
            /* Positional argument: a sentence */
            if (num_user_sentences >= MAX_SENTENCES) {
                fprintf(stderr, "Error: Too many sentences (max %d)\n", MAX_SENTENCES);
                return 1;
            }
            user_sentences[num_user_sentences++] = argv[i];
        }
    }

    /* Default sentences if none provided */
    const char *default_sentences[] = {
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
        "Machine learning is transforming industries.",
        "I love programming in C."
    };
    int num_default = sizeof(default_sentences) / sizeof(default_sentences[0]);

    /* Use user sentences or defaults */
    const char **sentences;
    int num_sentences;
    if (num_user_sentences > 0) {
        sentences = user_sentences;
        num_sentences = num_user_sentences;
    } else {
        sentences = default_sentences;
        num_sentences = num_default;
    }

    /* Load model */
    printf("Loading model from %s...\n", model_path);
    clock_t start = clock();

    gte_ctx *ctx = gte_load(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    double load_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Model loaded in %.2f seconds\n", load_time);
    printf("Embedding dimension: %d\n", gte_dim(ctx));
    printf("Max sequence length: %d\n\n", gte_max_seq_len(ctx));

    /* Generate embeddings */
    printf("Generating embeddings...\n\n");
    float **embeddings = malloc(num_sentences * sizeof(float *));

    for (int i = 0; i < num_sentences; i++) {
        start = clock();
        embeddings[i] = gte_embed(ctx, sentences[i]);
        double embed_time = (double)(clock() - start) / CLOCKS_PER_SEC;

        printf("S%d: \"%s\"\n", i + 1, sentences[i]);
        printf("    Time: %.3f ms\n", embed_time * 1000);
        printf("    Embedding: ");
        print_embedding(embeddings[i], gte_dim(ctx), 5);
        printf("\n");
    }

    /* Compute similarity matrix */
    printf("Cosine similarity matrix:\n");
    printf("     ");
    for (int i = 0; i < num_sentences; i++) {
        printf("  S%d   ", i + 1);
    }
    printf("\n");

    for (int i = 0; i < num_sentences; i++) {
        printf("S%d: ", i + 1);
        for (int j = 0; j < num_sentences; j++) {
            float sim = gte_cosine_similarity(embeddings[i], embeddings[j], gte_dim(ctx));
            printf(" %.3f ", sim);
        }
        printf("\n");
    }

    /* Cleanup */
    for (int i = 0; i < num_sentences; i++) {
        free(embeddings[i]);
    }
    free(embeddings);
    gte_free(ctx);

    printf("\nDone!\n");
    return 0;
}
