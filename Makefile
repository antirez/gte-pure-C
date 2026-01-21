CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm

# Detect OS for BLAS linking
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	BLAS_CFLAGS = -DACCELERATE_NEW_LAPACK
	BLAS_LDFLAGS = -framework Accelerate
else
	BLAS_CFLAGS = -I/usr/include -I/usr/local/include
	BLAS_LDFLAGS = -lblas
endif

all: generic

# Generic (pure C) build
generic: CFLAGS += -DUSE_GENERIC
generic: test_gte

# BLAS-accelerated build
blas: CFLAGS += -DUSE_BLAS $(BLAS_CFLAGS)
blas: LDFLAGS += $(BLAS_LDFLAGS)
blas: test_gte

test_gte: test_gte.c gte.c gte.h
	$(CC) $(CFLAGS) -o $@ test_gte.c gte.c $(LDFLAGS)

gte-small.gtemodel: offline/local_complete_model/model.safetensors
	python3 convert_model.py offline/local_complete_model $@

clean:
	rm -f test_gte bench

.PHONY: all clean generic blas
