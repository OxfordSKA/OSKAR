/* Copyright (c) 2011-2019, The University of Oxford. See LICENSE file. */

#define JONES_K_STATION 2
#define JONES_K_SOURCE 128

#define OSKAR_JONES_K_ARGS(FP, FP2)\
        const int       num_sources,\
        GLOBAL_IN(FP,   l),\
        GLOBAL_IN(FP,   m),\
        GLOBAL_IN(FP,   n),\
        const int       num_stations,\
        GLOBAL_IN(FP,   u),\
        GLOBAL_IN(FP,   v),\
        GLOBAL_IN(FP,   w),\
        const FP        wavenumber,\
        GLOBAL_IN(FP,   source_filter),\
        const FP        source_filter_min,\
        const FP        source_filter_max,\
        const int       ignore_w_components,\
        GLOBAL_OUT(FP2, jones)\

#define OSKAR_JONES_K_GPU(NAME, FP, FP2) KERNEL(NAME) (\
        OSKAR_JONES_K_ARGS(FP, FP2))\
{\
    const int s = GLOBAL_ID_X, a = GLOBAL_ID_Y;\
    const int ls = LOCAL_ID_X, la = LOCAL_ID_Y;\
    LOCAL FP f_[JONES_K_SOURCE];\
    LOCAL FP l_[JONES_K_SOURCE], m_[JONES_K_SOURCE], n_[JONES_K_SOURCE];\
    LOCAL FP u_[JONES_K_STATION], v_[JONES_K_STATION], w_[JONES_K_STATION];\
    if (s < num_sources && la == 0) {\
        l_[ls] = l[s];\
        m_[ls] = m[s];\
        n_[ls] = n[s] - (FP) 1;\
        f_[ls] = source_filter[s];\
    }\
    if (a < num_stations && ls == 0) {\
        u_[la] = wavenumber * u[a];\
        v_[la] = wavenumber * v[a];\
        w_[la] = wavenumber * w[a];\
    }\
    BARRIER;\
    FP2 weight; weight.x = weight.y = (FP) 0;\
    if (f_[ls] > source_filter_min && f_[ls] <= source_filter_max) {\
        FP re, im, phase;\
        phase  = u_[la] * l_[ls];\
        phase += v_[la] * m_[ls];\
        if (!ignore_w_components) phase += w_[la] * n_[ls];\
        SINCOS(phase, im, re);\
        weight.x = re; weight.y = im;\
    }\
    if (s < num_sources && a < num_stations)\
        jones[s + num_sources * a] = weight;\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_JONES_K_CPU(NAME, FP, FP2) KERNEL(NAME) (\
        OSKAR_JONES_K_ARGS(FP, FP2))\
{\
    KERNEL_LOOP_Y(int, a, 0, num_stations)\
    KERNEL_LOOP_X(int, s, 0, num_sources)\
    FP2 weight; weight.x = weight.y = (FP) 0;\
    if (source_filter[s] > source_filter_min &&\
                source_filter[s] <= source_filter_max) {\
        FP re, im, phase;\
        phase = u[a] * l[s] + v[a] * m[s];\
        if (!ignore_w_components) phase += w[a] * (n[s] - (FP)1);\
        phase *= wavenumber;\
        SINCOS(phase, im, re);\
        weight.x = re; weight.y = im;\
    }\
    jones[s + num_sources * a] = weight;\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
