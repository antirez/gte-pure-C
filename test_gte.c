/*
 * Test program for GTE library
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gte.h"

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
    const char *model_path = "gte-small.gtemodel";

    if (argc > 1) {
        model_path = argv[1];
    }

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

    /* Test sentences */
    const char *sentences[] = {
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
        "Machine learning is transforming industries.",
        "I love programming in C."
    };
    int num_sentences = sizeof(sentences) / sizeof(sentences[0]);

    /* Generate embeddings */
    printf("Generating embeddings...\n\n");
    float **embeddings = malloc(num_sentences * sizeof(float *));

    for (int i = 0; i < num_sentences; i++) {
        start = clock();
        embeddings[i] = gte_embed(ctx, sentences[i]);
        double embed_time = (double)(clock() - start) / CLOCKS_PER_SEC;

        printf("Sentence %d: \"%s\"\n", i + 1, sentences[i]);
        printf("  Time: %.3f ms\n", embed_time * 1000);
        printf("  First 5 dims: ");
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
