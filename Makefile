# Impostazione predefinita per il sistema
UNAME_S := $(shell uname -s)

# Configurazione per Ubuntu (GCC 13.3.0)
ifeq ($(UNAME_S), Linux)
  CC = gcc-13
  CFLAGS = -Iinclude -fopenmp -O0
  LDFLAGS = -fopenmp
  # Puoi anche aggiungere eventuali altre configurazioni specifiche per Ubuntu
endif

SRC = src/utils/mmio.c src/utils/read.c src/utils/write.c src/main.c src/implementations/csr.c src/implementations/ellpack.c 
OUT = project

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)

run: all
	./$(OUT)