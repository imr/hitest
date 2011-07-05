/* 
 * File:   main.cpp
 * Author: ian
 *
 * Created on June 26, 2011, 10:23 PM
 */

#include <mpi.h>
#include <omp.h>
#include <cuda.h>

float e;

// kernel
__global__ void sub1(float* fx, float* fy, float* fe) {
#define BLOCK (512)
  int t = threadIdx.x; // builtin
  int b = blockIdx.x; // builtin
  float e;
  __shared__ float se[BLOCK];
  __shared__ float sx[BLOCK];
  __shared__ float sy[BLOCK+2];
  // copy from device to processor memory
  sx[t] = fx[BLOCK*b+t];
  sy[t] = fy[BLOCK*b+t];
  if (t<2)
     sy[t+BLOCK] = fy[BLOCK*b+t+BLOCK];
  __syncthreads();

  // do computation
  sx[t] += ( sy[t+2] + sy[t] )*.5;
  e = sy[t+1] * sy[t+1];
  // copy to device memory
  fx[BLOCK*b+t] = sx[t];
  // reduction
  se[t] = e;
  __syncthreads();
  if (t<256) {
     se[t] += se[t+256];
     __syncthreads();
  }
  if (t<128) {
     se[t] += se[t+128];
     __syncthreads();
  }
  if (t<64) {
     se[t] += se[t+64];
     __syncthreads();
  }
  if (t<32) { // warp size
     se[t] += se[t+32];
     se[t] += se[t+16];
     se[t] += se[t+8];
     se[t] += se[t+4];
     se[t] += se[t+2];
     se[t] += se[t+1];
  }
  if (t==0)
     fe[b] = se[0];
}

int main(int argc, char *argv[]) {
  int n = 32;
  MPI_Init(&argc, &argv);
  int numproc, me;
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  int p_left = -1, p_right = -1;
  if (me > 0)
   p_left = me-1;
  if (me < numproc-1)
   p_right = me+1;
  int n_local0 = 1 + (me * (n-1)) / numproc;
  int n_local1 = 1 + ((me+1) * (n-1)) / numproc;
  // allocate only local part + ghost zone of the arrays x,y
  float *x, *y;
  x = new float[n_local1 - n_local0 + 2];
  y = new float[n_local1 - n_local0 + 2];
  x -= (n_local0 - 1);
  y -= (n_local0 - 1);

  // fill x, y

  // fill ghost zone
  MPI_Status s;
  if (p_left != -1)
    MPI_Send(&y[n_local0], 1, MPI_FLOAT, p_left,
      1, MPI_COMM_WORLD);
  if (p_right != -1) {
    MPI_Recv(&y[n_local1], 1, MPI_FLOAT, p_right,
      1, MPI_COMM_WORLD, &s);
    MPI_Send(&y[n_local1-1], 1, MPI_FLOAT, p_right,
      2, MPI_COMM_WORLD);
  }
  if (p_left != -1) 
    MPI_Recv(&y[n_local0-1], 1, MPI_FLOAT, p_left,
      2, MPI_COMM_WORLD, &s);
  

  e = 0;
  #pragma omp parallel
  {
  int p = omp_get_thread_num();
  int num = omp_get_num_threads();
  // pick GPU
  cudaSetDevice(p);
  // allocate GPU memory
  float *fx, *fy, *fe;
  cudaMalloc((void**)&fx, (n_local1-n_local0+2) * sizeof(float));
  cudaMalloc((void**)&fy, (n_local1-n_local0+2) * sizeof(float));
  cudaMalloc((void**)&fe, (n_local1-n_local0+2)/BLOCK * sizeof(float));
  float *de = new float[(n_local1-n_local0+2)/BLOCK];
  // copy to GPU memory
  cudaMemcpy(fx+1, &x[n_local0],
   (n_local1-n_local0) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(fy, &y[n_local0-1],
   (n_local1-n_local0+2) * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimBlock(BLOCK, 1, 1);
  dim3 dimGrid((n_local1-n_local0+2)/BLOCK, 1, 1);

  int n0 = 1+((n_local1-n_local0)*p)/num;
  int n1 = 1+((n_local1-n_local0)*(p+1))/num;
  // call GPU
  sub1<<<dimGrid, dimBlock>>>(fx, fy, fe);
  // copy to host memory
  cudaMemcpy(fx+1, &x[n0], (n1-n0) * sizeof(float),
   cudaMemcpyDeviceToHost);
  cudaMemcpy(fe, &de[n0-1], (n1-n0+2)/BLOCK * sizeof(float),
   cudaMemcpyDeviceToHost);
  // release GPU memory
  cudaFree(fe);
  cudaFree(fy);
  cudaFree(fx);
  float e_local = 0;
  for (int i=0; i<(n1-n0+2)/BLOCK; ++i)
   e_local += de[i];
  #pragma omp atomic
  e += e_local;
  delete[] de;
  }

  float e_local = e;
  MPI_Allreduce(&e_local, &e, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  // output x, e

  x += (n_local0 - 1);
  y += (n_local0 - 1);
  delete[] x, y;
  MPI_Finalize();
  return 0;
}
