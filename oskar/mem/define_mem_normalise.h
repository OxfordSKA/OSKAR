/* Copyright (c) 2019, The University of Oxford. See LICENSE file. */

#define OSKAR_MEM_NORMALISE_REAL_CPU(NAME, FP) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n,\
        GLOBAL_OUT(FP, a), const unsigned int idx)\
{\
    const FP scal = ((FP)1) / a[idx];\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    const unsigned int j = offset + i;\
    if (j != idx) a[j] *= scal;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_NORMALISE_COMPLEX_CPU(NAME, FP, FP2) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n,\
        GLOBAL_OUT(FP2, a), const unsigned int idx)\
{\
    const FP2 val = a[idx];\
    const FP amp = val.x * val.x + val.y * val.y;\
    const FP scal = RSQRT(amp);\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    const unsigned int j = offset + i;\
    if (j != idx) {\
        a[j].x *= scal; a[j].y *= scal;\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_NORMALISE_MATRIX_CPU(NAME, FP, FP4c) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n,\
        GLOBAL_OUT(FP4c, a), const unsigned int idx)\
{\
    const FP4c val = a[idx];\
    const FP amp = (val.a.x * val.a.x + val.a.y * val.a.y +\
            val.b.x * val.b.x + val.b.y * val.b.y +\
            val.c.x * val.c.x + val.c.y * val.c.y +\
            val.d.x * val.d.x + val.d.y * val.d.y) / (FP)2;\
    const FP scal = RSQRT(amp);\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    const unsigned int j = offset + i;\
    if (j != idx) {\
        a[j].a.x *= scal; a[j].a.y *= scal;\
        a[j].b.x *= scal; a[j].b.y *= scal;\
        a[j].c.x *= scal; a[j].c.y *= scal;\
        a[j].d.x *= scal; a[j].d.y *= scal;\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)


#define OSKAR_MEM_NORMALISE_REAL_GPU(NAME, FP) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n,\
        GLOBAL_OUT(FP, a), const unsigned int idx)\
{\
    const FP scal = ((FP)1) / a[idx];\
    const unsigned int i = GLOBAL_ID_X;\
    const unsigned int j = offset + i;\
    if (i < n && j != idx) a[j] *= scal;\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_NORMALISE_COMPLEX_GPU(NAME, FP, FP2) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n,\
        GLOBAL_OUT(FP2, a), const unsigned int idx)\
{\
    LOCAL FP scal;\
    if (LOCAL_ID_X == 0) {\
        const FP2 val = a[idx];\
        const FP amp = val.x * val.x + val.y * val.y;\
        scal = RSQRT(amp);\
    }\
    BARRIER;\
    const unsigned int i = GLOBAL_ID_X;\
    const unsigned int j = offset + i;\
    if (i < n && j != idx) {\
        a[j].x *= scal; a[j].y *= scal;\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_NORMALISE_MATRIX_GPU(NAME, FP, FP4c) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n,\
        GLOBAL_OUT(FP4c, a), const unsigned int idx)\
{\
    LOCAL FP scal;\
    if (LOCAL_ID_X == 0) {\
        const FP4c val = a[idx];\
        const FP amp = (val.a.x * val.a.x + val.a.y * val.a.y +\
                val.b.x * val.b.x + val.b.y * val.b.y +\
                val.c.x * val.c.x + val.c.y * val.c.y +\
                val.d.x * val.d.x + val.d.y * val.d.y) / (FP)2;\
        scal = RSQRT(amp);\
    }\
    BARRIER;\
    const unsigned int i = GLOBAL_ID_X;\
    const unsigned int j = offset + i;\
    if (i < n && j != idx) {\
        a[j].a.x *= scal; a[j].a.y *= scal;\
        a[j].b.x *= scal; a[j].b.y *= scal;\
        a[j].c.x *= scal; a[j].c.y *= scal;\
        a[j].d.x *= scal; a[j].d.y *= scal;\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)
