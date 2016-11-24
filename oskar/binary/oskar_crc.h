/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#ifndef OSKAR_CRC_H_
#define OSKAR_CRC_H_

/**
 * @file oskar_crc.h
 */

#include <binary/oskar_binary_macros.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OSKAR_CRC_TYPEDEF_
#define OSKAR_CRC_TYPEDEF_
typedef struct oskar_CRC oskar_CRC;
#endif /* OSKAR_CRC_TYPEDEF_ */

enum OSKAR_CRC_TYPE
{
    OSKAR_CRC_8_EBU,
    OSKAR_CRC_32,
    OSKAR_CRC_32C
};

/**
 * @brief
 * Creates the data table used for CRC computations.
 *
 * @details
 * Creates the data table used for CRC computations.
 *
 * The type is an enumerated value from the following list:
 *
 * - OSKAR_CRC_8_EBU
 * - OSKAR_CRC_32
 * - OSKAR_CRC_32C
 *
 * The CRC-8 scheme is from AES/EBU:
 * https://tech.ebu.ch/docs/tech/tech3250.pdf
 *
 * The standard CRC-32 is defined in IEEE 802.3 and is used by Ethernet
 * and PNG, among many others.
 *
 * The Castagnoli scheme, CRC-32C (used also in iSCSI, BTRFS, ext4)
 * has better error detection characteristics than IEEE 802.3:
 * http://dx.doi.org/10.1109/26.231911
 *
 * @param[in,out] type Enumerated CRC type.
 */
OSKAR_BINARY_EXPORT
oskar_CRC* oskar_crc_create(int type);

/**
 * @brief
 * Frees memory held by a CRC structure.
 *
 * @details
 * Frees memory held by a CRC structure.
 *
 * @param[in] data Pointer to data structure to free.
 */
OSKAR_BINARY_EXPORT
void oskar_crc_free(oskar_CRC* data);

/**
 * @brief
 * Updates a CRC value with new data.
 *
 * @details
 * Updates a CRC value with new data.
 *
 * Uses Intel's "slicing-by-8" algorithm for speed:
 * http://sourceforge.net/projects/slicing-by-8/
 * http://web.archive.org/web/20121011093914/http://www.intel.com/technology/comms/perfnet/download/CRC_generators.pdf
 * http://create.stephan-brumme.com/crc32/
 *
 * @param[in] crc_data  Pointer to CRC data table, which defines the type.
 * @param[in] crc       CRC code to update.
 * @param[in] data      Pointer to data block to use.
 * @param[in] num_bytes Length of data block in bytes.
 *
 * @return The updated CRC value.
 */
OSKAR_BINARY_EXPORT
unsigned long oskar_crc_update(const oskar_CRC* crc_data, unsigned long crc,
        const void* data, size_t num_bytes);

/**
 * @brief
 * Computes a CRC value from a block of memory.
 *
 * @details
 * Computes a CRC value from a block of memory by calling
 * oskar_crc_update().
 *
 * @param[in] crc_data  Pointer to CRC data table, which defines the type.
 * @param[in] data      Pointer to data block to use.
 * @param[in] num_bytes Length of data block in bytes.
 *
 * @return The computed CRC value.
 */
OSKAR_BINARY_EXPORT
unsigned long oskar_crc_compute(const oskar_CRC* crc_data, const void* data,
        size_t num_bytes);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CRC_H_ */
