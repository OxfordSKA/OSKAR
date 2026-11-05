/* Copyright (c) 2026, The OSKAR Developers. See LICENSE file. */

#define OSKAR_MEM_JONES_ROTATION(NAME, FP, FP4c) KERNEL(NAME) (\
        const int        offset,\
        const int        num_points,\
        const FP         cos_angle,\
        const FP         sin_angle,\
        GLOBAL_OUT(FP4c, jones))\
{\
    KERNEL_LOOP_X(int, i, 0, num_points)\
    const int idx = i + offset;\
    FP4c element = jones[idx];\
    /* Real. */\
    jones[idx].a.x = cos_angle * element.a.x - sin_angle * element.c.x;\
    jones[idx].b.x = cos_angle * element.b.x - sin_angle * element.d.x;\
    jones[idx].c.x = sin_angle * element.a.x + cos_angle * element.c.x;\
    jones[idx].d.x = sin_angle * element.b.x + cos_angle * element.d.x;\
    /* Imaginary. */\
    jones[idx].a.y = cos_angle * element.a.y - sin_angle * element.c.y;\
    jones[idx].b.y = cos_angle * element.b.y - sin_angle * element.d.y;\
    jones[idx].c.y = sin_angle * element.a.y + cos_angle * element.c.y;\
    jones[idx].d.y = sin_angle * element.b.y + cos_angle * element.d.y;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
