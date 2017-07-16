/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* oskar_mem_data_type_string(int data_type)
{
    switch (data_type)
    {
    case OSKAR_CHAR:                  return "CHAR";
    case OSKAR_INT:                   return "INT";
    case OSKAR_SINGLE:                return "SINGLE";
    case OSKAR_DOUBLE:                return "DOUBLE";
    case OSKAR_COMPLEX:               return "COMPLEX";
    case OSKAR_MATRIX:                return "MATRIX";
    case OSKAR_SINGLE_COMPLEX:        return "SINGLE COMPLEX";
    case OSKAR_DOUBLE_COMPLEX:        return "DOUBLE COMPLEX";
    case OSKAR_SINGLE_COMPLEX_MATRIX: return "SINGLE COMPLEX MATRIX";
    case OSKAR_DOUBLE_COMPLEX_MATRIX: return "DOUBLE COMPLEX MATRIX";
    default:                          break;
    };
    return "UNKNOWN TYPE";
}

#ifdef __cplusplus
}
#endif
