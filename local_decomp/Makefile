EXE_NAME = local_decompose
DEPS = GraphDecompose.h LMCache.h Graph.h Bigram.h WordHolder.h 
OBJ = GraphDecompose.o LMCache.o Graph.o Bigram.o WordHolder.o 
CC = g++ 
CFLAGS = -g
LIBS = -loolm -lmisc -ldstruct
INC = -I/home/nlg-03/riesa/tools/srilm/include/ -L/home/nlg-03/riesa/tools/srilm/lib/i686-m64/

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INC)


$(EXE_NAME): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(INC) $(LIBS)
