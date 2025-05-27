CC=clang++
CFLAGS=-g -Wall -std=c++17
DEPS=simplex_method.h constrained_optimization.h
OBJS=simplex_method.o solve_linear_programs.o
BIN=main

all:$(BIN)

main: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o main

%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) -r main *.o
