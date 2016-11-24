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

#ifndef OSKAR_VIS_HEADER_READ_H_
#define OSKAR_VIS_HEADER_READ_H_

/**
 * @file oskar_vis_header_read.h
 */

#include <oskar_global.h>
#include <binary/oskar_binary.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns a visibility structure by reading from the specified file.
 *
 * @details
 * This function returns a new visibility header structure by
 * reading from the specified file.
 *
 * The structure must be freed by calling oskar_vis_header_free()
 * when no longer required.
 *
 * @param[in]  h           Handle to an OSKAR binary file, opened for reading.
 * @param[in,out] status   Status return code.
 *
 * @return A handle to the new visibility header structure.
 */
OSKAR_EXPORT
oskar_VisHeader* oskar_vis_header_read(oskar_Binary* h, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VIS_HEADER_READ_H_ */
