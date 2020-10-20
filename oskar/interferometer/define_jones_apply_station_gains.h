/* Copyright (c) 2020, The OSKAR Developers. See LICENSE file. */

#define OSKAR_JONES_APPLY_STATION_GAINS_M(NAME, FP4c) KERNEL(NAME) (\
        const int        num_sources,\
        const int        num_stations,\
        GLOBAL_IN(FP4c,  gains),\
        GLOBAL_OUT(FP4c, jones))\
{\
    KERNEL_LOOP_Y(int, i_station, 0, num_stations)\
    const FP4c g = gains[i_station];\
    KERNEL_LOOP_X(int, i_source, 0, num_sources)\
    const int i_jones = num_sources * i_station + i_source;\
    const FP4c in = jones[i_jones];\
    FP4c out;\
    OSKAR_MUL_COMPLEX_MATRIX(out, g, in)\
    jones[i_jones] = out;\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_JONES_APPLY_STATION_GAINS_C(NAME, FP2) KERNEL(NAME) (\
        const int        num_sources,\
        const int        num_stations,\
        GLOBAL_IN(FP2,   gains),\
        GLOBAL_OUT(FP2,  jones))\
{\
    KERNEL_LOOP_Y(int, i_station, 0, num_stations)\
    const FP2 g = gains[i_station];\
    KERNEL_LOOP_X(int, i_source, 0, num_sources)\
    const int i_jones = num_sources * i_station + i_source;\
    const FP2 in = jones[i_jones];\
    FP2 out;\
    OSKAR_MUL_COMPLEX(out, g, in)\
    jones[i_jones] = out;\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
