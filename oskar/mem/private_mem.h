/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_MEM_H_
#define OSKAR_PRIVATE_MEM_H_

#include <stddef.h> /* For size_t */

#include <oskar_global.h>
#include <utility/oskar_thread.h>

#ifdef OSKAR_HAVE_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#endif

struct oskar_Mem
{
    int type;            /* Enumerated element type of memory block. */
    int location;        /* Enumerated address space of data pointer. */
    size_t num_elements; /* Number of elements in memory block. */
    int owner;           /* Flag set if the structure owns the memory. */
    void* data;          /* Data pointer. */

#ifdef OSKAR_HAVE_OPENCL
    cl_mem buffer;       /* Handle to OpenCL buffer. */
#endif

    int ref_count;       /* Reference counter. */
    oskar_Mutex* mutex;  /* Mutex to guard reference counter. */
};

#ifndef OSKAR_MEM_TYPEDEF_
#define OSKAR_MEM_TYPEDEF_
typedef struct oskar_Mem oskar_Mem;
#endif /* OSKAR_MEM_TYPEDEF_ */

#endif /* include guard */
