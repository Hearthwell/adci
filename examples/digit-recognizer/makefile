C_FLAGS:=-Wall -Wextra -g

C_COMMON_FLAGS:=-I../../include

all: main.o weights.o
	gcc $(C_FLAGS) $^ -o main -L../../ -ladci -lm

weights.o:../digit-recognizer-nograph/weights.c
	gcc $(C_FLAGS) $(C_COMMON_FLAGS) -c $< -o $@

%.o: %.c
	gcc $(C_FLAGS) $(C_COMMON_FLAGS) -c $< -o $@

clean:
	rm -rf main
	rm -rf *.o