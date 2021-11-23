/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_mem_read_binary_raw(const char* filename, int type,
        int location, int* status)
{
    size_t num_elements = 0, element_size = 0, size_bytes = 0;
    oskar_Mem *mem = 0;
    FILE* stream = 0;
    if (*status) return 0;

    /* Open the input file. */
    stream = fopen(filename, "rb");
    if (!stream)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Get the file size. */
    fseek(stream, 0, SEEK_END);
    size_bytes = ftell(stream);

    /* Create memory block of the right size. */
    element_size = oskar_mem_element_size(type);
    num_elements = (size_t)ceil(size_bytes / element_size);
    mem = oskar_mem_create(type, OSKAR_CPU, num_elements, status);
    if (*status)
    {
        oskar_mem_free(mem, status);
        fclose(stream);
        return 0;
    }

    /* Read the data. */
    fseek(stream, 0, SEEK_SET);
    if (fread(oskar_mem_void(mem), 1, size_bytes, stream) != size_bytes)
    {
        oskar_mem_free(mem, status);
        fclose(stream);
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Close the input file. */
    fclose(stream);

    /* Copy to GPU memory if required. */
    if (location != OSKAR_CPU)
    {
        oskar_Mem* gpu = 0;
        gpu = oskar_mem_create_copy(mem, location, status);
        oskar_mem_free(mem, status);
        return gpu;
    }

    return mem;
}

#ifdef __cplusplus
}
#endif
