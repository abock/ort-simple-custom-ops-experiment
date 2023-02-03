ORT_ROOT_DIR := $(HOME)/src/onnxruntime/
ORT_BUILD_CONFIG := RelWithDebInfo

ORT_INCLUDE_DIR := $(ORT_ROOT_DIR)include/
ORT_LIB_DIR := $(ORT_ROOT_DIR)build/Linux/$(ORT_BUILD_CONFIG)/

CFLAGS := -Wall -Werror -ggdb3 -I$(ORT_INCLUDE_DIR)
LDFLAGS := -L$(ORT_LIB_DIR) -lm -lonnxruntime

APPBINARY := custom-op-demo
OBJECTS := $(APPBINARY).o new-custom-op-api.o

.PHONY: all
all: $(APPBINARY)

.PHONY: run
run: $(APPBINARY)
	LD_LIBRARY_PATH=$(ORT_LIB_DIR) gdb -ex run ./$<

$(APPBINARY): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $+

$(OBJECTS): %.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY:
clean:
	rm -f $(OBJECTS) $(APPBINARY)
