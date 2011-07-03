/* 
 * File:   main.cpp
 * Author: Ian Roth
 * License: Simplifed BSD License
 *
 * Created on June 26, 2011, 10:23 PM
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "kernel.cu"

using namespace std;

char processor_name[MPI_MAX_PROCESSOR_NAME];
bool initDevice();
extern "C" void kernel();
bool initDevice()
{
    printf("Init device %d on %s\n",omp_get_thread_num(),processor_name);
    return (cudaSetDevice(omp_get_thread_num()) == cudaSuccess);
}

int main(int argc, char* argv[])
{
    int numprocs,namelen,rank,devCount;
    int val = 0;
    MPI_Status stat;

    // Initialize MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);
    printf("Hello from %d on %s out of %d\n",(rank+1),processor_name,numprocs);
    if (cudaGetDeviceCount(&devCount) != cudaSuccess)
    {
        printf("Device error on %s\n!",processor_name);
        MPI_Finalize();
        return 1;
    }

    // Test MPI message passing
    if (rank == 0){
        val = 3;
        for (int i=0; i<numprocs; i++)
            MPI_Send(&val,1,MPI_INT,i,0,MPI_COMM_WORLD);
    }
    MPI_Recv(&val,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat);
    if (val == 3)
        cout << rank << " properly received via MPI!" << endl;
    else
        cout << rank << " had an error receiving over MPI!" << endl;

    // Run one OpenMP thread per device per MPI node
    #pragma omp parallel num_threads(devCount)
    if (initDevice())
    {
        // Block and grid dimensions
        dim3 dimBlock(12,12);
        //kernel<<<1,dimBlock>>>();
        cudaThreadExit();
    }
    else
    {
        printf("Device error on %s\n",processor_name);
    }
    MPI_Finalize();
    return 0;
}


