CC=/usr/local/cuda/bin/nvcc
CFLAGS= -I/usr/lib/openmpi/include -I/usr/local/cuda/include -Xcompiler -fopenmp
LDFLAGS= -L/usr/lib/openmpi/lib -L/usr/local/cuda/lib
LIB= -lgomp -lcudart -lmpi
SOURCES= main.cpp
EXECNAME= hitest

all:
	$(CC) -o $(EXECNAME) $(SOURCES) $(LIB) $(LDFLAGS) $(CFLAGS)

clean:
	rm *.o *.linkinfo

