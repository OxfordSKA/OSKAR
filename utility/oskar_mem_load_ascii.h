/*
 * Copyright (c) 2013-2014, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_MEM_LOAD_ASCII_H_
#define OSKAR_MEM_LOAD_ASCII_H_

/**
 * @file oskar_mem_load_ascii.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads an ASCII table from file to populate the given blocks of memory.
 *
 * @details
 * This function reads an ASCII table and populates the supplied arrays
 * from columns in the table.
 *
 * The variable argument list must contain, for each array, a pointer to an
 * oskar_Mem structure, and a string containing the default value for elements
 * in that array. These parameters alternate throughout the list, so they would
 * appear as: oskar_Mem*, const char*, oskar_Mem*, const char* ...
 *
 * If the default is a blank string, then it is a required column.
 *
 * Note that, for a single column file (with no default), the default must
 * be passed as a blank string, i.e. "".
 *
 * Data within oskar_Mem structures may reside either in CPU or GPU memory.
 * The number of structures passed is given by the \p num_mem parameter.
 *
 * @param[in] filename      Pathname of file to read.
 * @param[in] num_mem       Number of arrays passed to this function.
 * @param[in,out]  status   Status return code.
 *
 * @return The number of rows read from the file.
 */
OSKAR_EXPORT
size_t oskar_mem_load_ascii(const char* filename, size_t num_mem,
        int* status, ...);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_LOAD_ASCII_H_ */
