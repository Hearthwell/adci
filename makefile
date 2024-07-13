SRC_DIR:=src
OUT_DIR:=out
PROJECT_NAME:=adci

EXTERNAL_SRC_FILES:=$(wildcard $(SRC_DIR)/external/*.c)
SRC_FILES:=$(wildcard $(SRC_DIR)/*.c)

EXTERNAL_OBJ:=$(EXTERNAL_SRC_FILES:$(SRC_DIR)/external/%.c=$(OUT_DIR)/%.c.o)
SRC_OBJ:=$(SRC_FILES:$(SRC_DIR)/%.c=$(OUT_DIR)/%.c.o) $(EXTERNAL_OBJ)
SRC_OBJ_REL:=$(SRC_FILES:$(SRC_DIR)/%.c=$(OUT_DIR)/%.c.rel.o) $(EXTERNAL_OBJ)

CC:=gcc
C_DEBUG_FLAGS:=-DADCI_BUILD_DEBUG -g
C_RELEASE_FLAGS:=-O3
C_COMMON_FLAGS:=-Wall -Wextra -Iinclude -Isrc
C_LIBS:=-lm

$(OUT_DIR)/%.c.o:$(SRC_DIR)/external/%.c
	$(CC) $(C_RELEASE_FLAGS) -c -o $@ $<

$(OUT_DIR)/%.c.rel.o:$(SRC_DIR)/%.c
	$(CC) $(C_COMMON_FLAGS) $(C_RELEASE_FLAGS) -c -o $@ $<

$(OUT_DIR)/%.c.o:$(SRC_DIR)/%.c
	$(CC) $(C_COMMON_FLAGS) $(C_DEBUG_FLAGS) $(C_FLAGS) -c -o $@ $<

release_lib: $(SRC_OBJ_REL) 
	ar rcs lib$(PROJECT_NAME).a $^

lib: $(SRC_OBJ)
	ar rcs lib$(PROJECT_NAME)dbg.a $^

TEST_DIR:=test
TEST_FILES:=$(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ:=$(TEST_FILES:$(TEST_DIR)/%.cpp=$(OUT_DIR)/%.cpp.o)
$(OUT_DIR)/%.cpp.o:$(TEST_DIR)/%.cpp
	g++ $(C_DEBUG_FLAGS) $(C_COMMON_FLAGS) -c -o $@ $<
tests: $(SRC_OBJ) $(TEST_OBJ) 
	g++ $^ -o $@ -lgtest

clean:
	rm -rf $(OUT_DIR)/*
	rm -f main tests
	rm -rf *.a