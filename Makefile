CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm

all: test_gte

test_gte: test_gte.c gte.c gte.h
	$(CC) $(CFLAGS) -o $@ test_gte.c gte.c $(LDFLAGS)

gte-small.gtemodel: offline/local_complete_model/model.safetensors
	python3 convert_model.py offline/local_complete_model $@

clean:
	rm -f test_gte bench

.PHONY: all clean
