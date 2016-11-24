/*
 * Copyright (c) 2014, The University of Oxford
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

#include "binary/oskar_crc.h"
#include "binary/oskar_endian.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_CRC
{
    int type;
    unsigned long poly;
    unsigned long init;
    unsigned long xorout;
    unsigned long t[8][256];
};
#ifndef OSKAR_CRC_TYPEDEF_
#define OSKAR_CRC_TYPEDEF_
typedef struct oskar_CRC oskar_CRC;
#endif /* OSKAR_CRC_TYPEDEF_ */


oskar_CRC* oskar_crc_create(int type)
{
    int i, j;
    oskar_CRC* d;

    /* Create the data structure. */
    d = (oskar_CRC*) malloc(sizeof(oskar_CRC));
    d->type = type;

    /* Set the polynomial, initial and post-XOR values based on type. */
    /* Always need the "reversed" form of the polynomial for this generator. */
    if (type == OSKAR_CRC_8_EBU)
    {
        d->poly   = 0xb8;
        d->init   = 0xFF;
        d->xorout = 0;
    }
    else if (type == OSKAR_CRC_32)
    {
        d->poly   = 0xedb88320uL;
        d->init   = 0xFFFFFFFFuL;
        d->xorout = 0xFFFFFFFFuL;
    }
    else if (type == OSKAR_CRC_32C)
    {
        d->poly   = 0x82f63b78uL;
        d->init   = 0xFFFFFFFFuL;
        d->xorout = 0xFFFFFFFFuL;
    }

    /* Fill the lookup table, starting with standard Sarwate CRC algorithm. */
    for (i = 0; i <= 0xFF; i++)
    {
        unsigned long crc;
        crc = (unsigned long) i;
        for (j = 0; j < 8; j++)
        {
            crc = (crc >> 1) ^ ((crc & 1) * d->poly);
        }
        d->t[0][i] = crc;
    }

    /* Then use Intel's "slicing-by-8" algorithm for speed:
     * http://sourceforge.net/projects/slicing-by-8/
     * http://web.archive.org/web/20121011093914/http://www.intel.com/technology/comms/perfnet/download/CRC_generators.pdf
     * http://create.stephan-brumme.com/crc32/
     */
    for (i = 0; i <= 0xFF; i++)
    {
        for (j = 0; j < 7; j++)
        {
            d->t[j + 1][i] = (d->t[j][i] >> 8) ^ d->t[0][d->t[j][i] & 0xFF];
        }
    }

    return d;
}

void oskar_crc_free(oskar_CRC* data)
{
    free(data);
}

unsigned long oskar_crc_update(const oskar_CRC* crc_data, unsigned long crc,
        const void* data, size_t num_bytes)
{
    const unsigned char* byte;
    unsigned char d[8];

    /* Use 8-byte chunks. */
    if (crc != crc_data->init) crc ^= crc_data->xorout;
    byte = (const unsigned char*) data;
    if (oskar_endian() == OSKAR_LITTLE_ENDIAN)
    {
        while (num_bytes >= 8)
        {
            num_bytes -= 8;
            memcpy(d, byte, 8);
            byte += 8;
            d[0] ^= crc         & 0xFF;
            d[1] ^= (crc >> 8)  & 0xFF;
            d[2] ^= (crc >> 16) & 0xFF;
            d[3] ^= (crc >> 24) & 0xFF;
            crc =   crc_data->t[0][d[7]] ^ crc_data->t[1][d[6]] ^
                    crc_data->t[2][d[5]] ^ crc_data->t[3][d[4]] ^
                    crc_data->t[4][d[3]] ^ crc_data->t[5][d[2]] ^
                    crc_data->t[6][d[1]] ^ crc_data->t[7][d[0]];
        }
    }
    else
    {
        while (num_bytes >= 8)
        {
            num_bytes -= 8;
            memcpy(d, byte, 8);
            byte += 8;
            d[0] ^= (crc >> 24) & 0xFF;
            d[1] ^= (crc >> 16) & 0xFF;
            d[2] ^= (crc >> 8)  & 0xFF;
            d[3] ^= crc         & 0xFF;
            crc =   crc_data->t[0][d[4]] ^ crc_data->t[1][d[5]] ^
                    crc_data->t[2][d[6]] ^ crc_data->t[3][d[7]] ^
                    crc_data->t[4][d[0]] ^ crc_data->t[5][d[1]] ^
                    crc_data->t[6][d[2]] ^ crc_data->t[7][d[3]];
        }
    }

    /* Must do remaining bytes individually. */
    while (num_bytes--)
        crc = (crc >> 8) ^ crc_data->t[0][(crc & 0xFF) ^ *byte++];

    return crc ^ crc_data->xorout;
}

unsigned long oskar_crc_compute(const oskar_CRC* crc_data, const void* data,
        size_t num_bytes)
{
    return oskar_crc_update(crc_data, crc_data->init, data, num_bytes);
}

#ifdef __cplusplus
}
#endif
