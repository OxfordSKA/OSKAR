/* Copyright (c) 2018-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_GRID_CORRECTION(NAME, FP) KERNEL(NAME) (\
        const int image_size, GLOBAL_IN(FP, corr_func),\
        GLOBAL_OUT(FP, complex_image))\
{\
    KERNEL_LOOP_Y(int, j, 0, image_size)\
    KERNEL_LOOP_X(int, i, 0, image_size)\
    const FP t = corr_func[i] * corr_func[j];\
    complex_image[((j * image_size + i) << 1)] *= t;\
    complex_image[((j * image_size + i) << 1) + 1] *= t;\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
