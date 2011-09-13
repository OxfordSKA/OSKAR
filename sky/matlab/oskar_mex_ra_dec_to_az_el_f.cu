#include <mex.h>

#include "sky/oskar_cuda_ra_dec_to_az_el.h"

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void mexFunction(int /*num_outputs*/, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    // Parse Inputs.
    if (num_inputs != 4)
        mexErrMsgTxt("Four inputs required ==> (ra_rad, dec_rad, lst_rad, lat_rad)");

    // Get matlab inputs.
    float ra  = (float) mxGetScalar(input[0]);
    float dec = (float) mxGetScalar(input[1]);
    float lst = (float) mxGetScalar(input[2]);
    float lat = (float) mxGetScalar(input[3]);

    mexPrintf("- ra  = %f\n", ra);
    mexPrintf("- dec = %f\n", dec);
    mexPrintf("- lst = %f\n", lst);
    mexPrintf("- lat = %f\n", lat);

    int n = 1;
    mwSize n_dims  = 1;
    mwSize dims[1] = {n};
    output[0] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    output[1] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    float* az = (float*) mxGetPr(output[0]);
    float* el = (float*) mxGetPr(output[1]);
    size_t mem_size = n * sizeof(float);
    float* d_ra;
    cudaMalloc((void**)&d_ra, mem_size);
    cudaMemcpy(d_ra, &ra, mem_size, cudaMemcpyHostToDevice);
    float* d_dec;
    cudaMalloc((void**)&d_dec, mem_size);
    cudaMemcpy(d_dec, &dec, mem_size, cudaMemcpyHostToDevice);
    float* d_az;
    cudaMalloc((void**)&d_az, mem_size);
    float* d_el;
    cudaMalloc((void**)&d_el, mem_size);
    float* d_work;
    cudaMalloc((void**)&d_work, mem_size);

    int error = (int)cudaSuccess;
    error = oskar_cuda_ra_dec_to_az_el_f(n, d_ra, d_dec, lst, lat, d_work,
            d_az, d_el);
    cudaMemcpy(az, d_az, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(el, d_el, mem_size, cudaMemcpyDeviceToHost);

    mexPrintf("- az = %f\n", az);
    mexPrintf("- el = %f\n", el);

    cudaFree(d_ra);
    cudaFree(d_dec);
    cudaFree(d_az);
    cudaFree(d_el);
    cudaFree(d_work);

    if (error != cudaSuccess)
    {
        mexPrintf("****************************************************\n");
        mexPrintf("** CUDA ERROR[%i]: %s.\n", error,
                cudaGetErrorString((cudaError_t)error));
        mexPrintf("****************************************************\n");
        mexErrMsgTxt("** ERROR: oskar_ra_dec_to_az_el_f()");
    }
}
