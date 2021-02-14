CC = g++
SRCS = $(wildcard ./*.cpp)
OBJS = $(SRCS:.cpp=.o)
PROG = lr
CFLAGS = -O2 -Wall -Wextra -g

all: $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

.cpp.o: 
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	rm -rf ./*.o $(PROG)
