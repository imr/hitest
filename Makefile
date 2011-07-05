#
CC=/opt/nvidia/cuda/bin/nvcc
CFLAGS= -I/usr/include/openmpi-x86_64 -I/opt/nvidia/cuda/include 
LDFLAGS= -L/usr/lib64/openmpi/lib -L/opt/nvidia/cuda/lib
LIB= -lgomp -lcuda -lmpi
SOURCES= main.cu
EXECNAME= hitest

all:
	$(CC) -o $(EXECNAME) $(SOURCES) $(LIB) $(LDFLAGS) $(CFLAGS)

clean:
	rm *.o *.linkinfo
