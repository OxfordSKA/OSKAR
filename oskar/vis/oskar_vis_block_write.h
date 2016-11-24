/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_VIS_BLOCK_WRITE_H_
#define OSKAR_VIS_BLOCK_WRITE_H_

/**
 * @file oskar_vis_block_write.h
 */

#include <oskar_global.h>
#include <binary/oskar_binary.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Write a visibility structure to the specified file.
 *
 * @details
 * This function writes a visibility structure to the specified file handle.
 *
 * @param[in,out] vis         The visibility block structure to write.
 * @param[in,out] h           The OSKAR binary file handle, opened for write.
 * @param[in]     block_index The visibility block index.
 * @param[in,out] status      Status return code.
 */
OSKAR_EXPORT
void oskar_vis_block_write(const oskar_VisBlock* vis, oskar_Binary* h,
        int block_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VIS_BLOCK_WRITE_H_ */
