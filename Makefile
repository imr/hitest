#
CC=/usr/local/cuda/bin/nvcc
CFLAGS= -I/usr/lib64/openmpi/1.4-gcc/include -I/usr/local/cuda/include 
LDFLAGS= -L/usr/lib64/openmpi/1.4-gcc/lib -L/usr/local/cuda/lib
LIB= -lgomp -lcuda -lmpi
SOURCES= main.cu
EXECNAME= hitest

all:
	$(CC) -o $(EXECNAME) $(SOURCES) $(LIB) $(LDFLAGS) $(CFLAGS)

clean:
	rm *.o *.linkinfo
