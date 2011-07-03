/*
 * File:   kernel.cu
 * Author: Ian Roth
 * License: Simplifed BSD License
 *
 * Created on June 26, 2011, 10:23 PM
 */

#ifndef _BURN_KERNEL_H_
#define _BURN_KERNEL_H_
extern "C"
{
    __global__ void kernel()
    {
        __shared__ float shared[512];
        float a = 3.0 * 5.0;
        float b = (a * 50) / 4;
        //int pos = threadIdx.y*blockDim.x+threadIdx.x;
        //shared[pos] = b;
    }
}
#endif

