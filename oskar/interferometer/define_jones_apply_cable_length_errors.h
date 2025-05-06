/* Copyright (c) 2025, The OSKAR Developers. See LICENSE file. */

#define OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_M(NAME, FP, FP4c) KERNEL(NAME) (\
        const int        num_sources,\
        const int        num_stations,\
        const FP         wavenumber,\
        const int        apply_x,\
        const int        apply_y,\
        GLOBAL_IN(FP,    errors_x),\
        GLOBAL_IN(FP,    errors_y),\
        GLOBAL_OUT(FP4c, jones))\
{\
    KERNEL_LOOP_Y(int, i_station, 0, num_stations)\
    FP4c gain;\
    FP re, im, phase;\
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, gain);\
    if (apply_x) {\
        phase = wavenumber * errors_x[i_station];\
        SINCOS(phase, im, re);\
        gain.a.x = re; gain.a.y = im;\
    } else {\
        gain.a.x = (FP) 1;\
    }\
    if (apply_y) {\
        phase = wavenumber * errors_y[i_station];\
        SINCOS(phase, im, re);\
        gain.d.x = re; gain.d.y = im;\
    } else {\
        gain.d.x = (FP) 1;\
    }\
    KERNEL_LOOP_X(int, i_source, 0, num_sources)\
    const int i_jones = num_sources * i_station + i_source;\
    const FP4c in = jones[i_jones];\
    FP4c out;\
    OSKAR_MUL_COMPLEX_MATRIX(out, gain, in)\
    jones[i_jones] = out;\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_C(NAME, FP, FP2) KERNEL(NAME) (\
        const int        num_sources,\
        const int        num_stations,\
        const FP         wavenumber,\
        const int        apply_x,\
        const int        apply_y,\
        GLOBAL_IN(FP,    errors_x),\
        GLOBAL_IN(FP,    errors_y),\
        GLOBAL_OUT(FP2,  jones))\
{\
    (void) apply_y;\
    (void) errors_y;\
    if (!apply_x) return;\
    KERNEL_LOOP_Y(int, i_station, 0, num_stations)\
    FP2 gain;\
    FP re, im;\
    const FP phase = wavenumber * errors_x[i_station];\
    SINCOS(phase, im, re);\
    gain.x = re; gain.y = im;\
    KERNEL_LOOP_X(int, i_source, 0, num_sources)\
    const int i_jones = num_sources * i_station + i_source;\
    const FP2 in = jones[i_jones];\
    FP2 out;\
    OSKAR_MUL_COMPLEX(out, gain, in)\
    jones[i_jones] = out;\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
