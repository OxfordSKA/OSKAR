/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_MEM_BINARY_FILE_READ_H_
#define OSKAR_MEM_BINARY_FILE_READ_H_

/**
 * @file oskar_mem_binary_file_read.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_BinaryTag.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads an OSKAR memory block from an OSKAR binary file.
 *
 * @details
 * This function loads the contents of an OSKAR memory block from a binary file.
 *
 * @param[in] mem          Pointer to data structure.
 * @param[in] filename     Name of binary file.
 * @param[in,out] index    Pointer to an index structure pointer.
 * @param[in] data_type    Type of the memory (as in oskar_Mem).
 * @param[in] name_group   Tag group name.
 * @param[in] name_tag     Tag name.
 * @param[in] user_index   User-defined index.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A positive return code indicates a CUDA error.
 * - A negative return code indicates an OSKAR error.
 */
OSKAR_EXPORT
int oskar_mem_binary_file_read(oskar_Mem* mem, const char* filename,
        oskar_BinaryTagIndex** index, const char* name_group,
        const char* name_tag, int user_index);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_BINARY_FILE_READ_H_ */
