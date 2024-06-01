SRC_DIR:=src
OUT_DIR:=out
EXAMPLES_DIR:=examples

SRC_FILES:=$(wildcard $(SRC_DIR)/*.c)
SRC_OBJ:=$(SRC_FILES:$(SRC_DIR)/%.c=$(OUT_DIR)/%.c.o)

CC:=gcc
C_FLAGS:=-Wall -Wextra -Iinclude
C_LIBS:=

$(OUT_DIR)/%.c.o:$(SRC_DIR)/%.c
	$(CC) $(C_FLAGS) -c -o $@ $<

$(OUT_DIR)/%.c.o:$(EXAMPLES_DIR)/%.c
	$(CC) $(C_FLAGS) -c -o $@ $<

main: $(SRC_OBJ) $(OUT_DIR)/main.c.o
	$(CC) $^ -o $@ $(C_LIBS)

tests: $(SRC_OBJ)
	@echo "TODO, IMPLEMENT"

clean:
	rm -rf $(OUT_DIR)/*
	rm -f main tests