/* Copyright (c) 2017-2019, The University of Oxford. See LICENSE file. */

/* Launch 1 thread block with 512 (or max number of) threads.
 * Shared memory size is num_threads * 2 * sizeof(T). */
#define OSKAR_PREFIX_SUM_GPU(NAME, T) KERNEL(NAME) (const int num_elements,\
        GLOBAL_IN(T, in), GLOBAL_OUT(T, out) LOCAL_CL(T, scratch))\
{\
    LOCAL_CUDA_BASE(T, scratch)\
    const int tid = LOCAL_ID_X, bdim = LOCAL_DIM_X;\
    const int num_loops = (num_elements + bdim) / bdim;\
    T running_total = (T)0;\
    int idx = tid; /* Starting value. */\
    const int t = tid + bdim;\
    for (int i = 0; i < num_loops; i++) {\
        T val = (T)0;\
        if (idx <= num_elements && idx > 0)\
            val = in[idx - 1];\
        scratch[tid] = (T)0; scratch[t] = val;\
        for (int j = 1; j < bdim; j <<= 1) {\
            BARRIER; const T x = scratch[t - j];\
            BARRIER; scratch[t] += x;\
        }\
        /* Store results. Note the very last element is the total number! */\
        BARRIER;\
        if (idx <= num_elements)\
            out[idx] = scratch[t] + running_total;\
        idx += bdim;\
        running_total += scratch[2 * bdim - 1];\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_PREFIX_SUM_CPU(NAME, T) KERNEL(NAME) (const int num_elements,\
        GLOBAL_IN(T, in), GLOBAL_OUT(T, out))\
{\
    if (GLOBAL_ID_X == 0) {\
        int i;\
        T sum = (T)0;\
        for (i = 0; i < num_elements; ++i) {\
            T x = in[i]; out[i] = sum; sum += x;\
        }\
        out[i] = sum;\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)
