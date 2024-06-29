SRC_DIR:=src
OUT_DIR:=out
EXAMPLES_DIR:=examples

SRC_FILES:=$(wildcard $(SRC_DIR)/*.c)
SRC_OBJ:=$(SRC_FILES:$(SRC_DIR)/%.c=$(OUT_DIR)/%.c.o)

CC:=gcc
# TODO, REMOVE DEBUG INFO / ADD RELEASE PRESET, BUILD
C_DEBUG_FLAGS:=-DADCI_BUILD_DEBUG -g
C_FLAGS:=-Wall -Wextra -Iinclude -Isrc $(C_DEBUG_FLAGS)
C_LIBS:=

$(OUT_DIR)/%.c.o:$(SRC_DIR)/%.c
	$(CC) $(C_FLAGS) -c -o $@ $<

$(OUT_DIR)/%.c.o:$(EXAMPLES_DIR)/%.c
	$(CC) $(C_FLAGS) -c -o $@ $<

main: $(SRC_OBJ) $(OUT_DIR)/main.c.o
	$(CC) $^ -o $@ $(C_LIBS)

TEST_DIR:=test
TEST_FILES:=$(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ:=$(TEST_FILES:$(TEST_DIR)/%.cpp=$(OUT_DIR)/%.cpp.o)
$(OUT_DIR)/%.cpp.o:$(TEST_DIR)/%.cpp
	g++ $(C_FLAGS) -c -o $@ $<
tests: $(SRC_OBJ) $(TEST_OBJ) 
	g++ $^ -o $@ -lgtest

clean:
	rm -rf $(OUT_DIR)/*
	rm -f main tests