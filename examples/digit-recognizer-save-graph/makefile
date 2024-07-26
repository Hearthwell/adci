C_FLAGS:=-Wall -Wextra -g

C_COMMON_FLAGS:=-I../../include

all: save_graph.o weights.o main.o
# TODO, PUT BACK DEBUG LIB
	gcc $(C_FLAGS) main.o -o main -L../../ -ladcidbg -lm
	gcc $(C_FLAGS) save_graph.o weights.o -o save_graph -L../../ -ladci -lm

weights.o:../digit-recognizer-nograph/weights.c
	gcc $(C_FLAGS) $(C_COMMON_FLAGS) -c $< -o $@

%.o: %.c
	gcc $(C_FLAGS) $(C_COMMON_FLAGS) -c $< -o $@

clean:
	rm -rf main save_graph
	rm -rf *.o