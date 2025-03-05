# Impostazione predefinita per il sistema
UNAME_S := $(shell uname -s)

# Configurazione per Ubuntu (GCC 13.3.0)
ifeq ($(UNAME_S), Linux)
  CC = gcc-13
  CFLAGS = -Iinclude -Wall -Wextra -fopenmp -O0
  LDFLAGS = -fopenmp
  # Puoi anche aggiungere eventuali altre configurazioni specifiche per Ubuntu
endif

SRC = src/read.c src/read.c src/write.c
OUT = progetto.out

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)

run: all
	./$(OUT)