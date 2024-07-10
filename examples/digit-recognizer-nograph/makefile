C_FLAGS:=-Wall -Wextra -O3

%.o: %.c
	gcc $(C_FLAGS) -I../../include -c $< -o $@

all: main.o weights.o
	gcc $(C_FLAGS) -I../../include $^ -o main -L../../ -ladci -lm

clean:
	rm -rf main
	rm -rf *.o