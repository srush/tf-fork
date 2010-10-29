EXE_NAME = test
DEPS = GraphDecompose.h LMCache.h Graph.h Bigram.h WordHolder.h 
OBJ = dual_subproblem.o  Test.o GraphDecompose.o LMCache.o Graph.o Bigram.o WordHolder.o  
CC = g++ 
CFLAGS = -g
LIBS = -loolm -lmisc -ldstruct -lcpptest
#INC = -I/home/nlg-03/riesa/tools/srilm/include/ -L/home/nlg-03/riesa/tools/srilm/lib/i686-m64/
INC = -I/home/srush/libs/sri/include/ -L/home/srush/libs/sri/lib/i686/

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INC)


$(EXE_NAME): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(INC) $(LIBS)
