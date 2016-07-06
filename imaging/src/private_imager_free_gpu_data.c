/*
 * Copyright (c) 2016, The University of Oxford
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

#include <oskar_device_utils.h>
#include <private_imager.h>
#include <private_imager_free_gpu_data.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_free_gpu_data(oskar_Imager* h, int* status)
{
    int i;
    for (i = 0; i < h->num_gpus && h->d; ++i)
    {
        DeviceData* dd = &h->d[i];
        if (!dd) continue;
        oskar_device_set(h->cuda_device_ids[i], status);
        oskar_mem_free(dd->uu, status);
        oskar_mem_free(dd->vv, status);
        oskar_mem_free(dd->ww, status);
        oskar_mem_free(dd->amp, status);
        oskar_mem_free(dd->weight, status);
        oskar_mem_free(dd->l, status);
        oskar_mem_free(dd->m, status);
        oskar_mem_free(dd->n, status);
        oskar_mem_free(dd->block_gpu, status);
        oskar_mem_free(dd->block_cpu, status);
        oskar_device_reset();
    }
    free(h->d);
    free(h->cuda_device_ids);
    h->cuda_device_ids = 0;
    h->num_gpus = 0;
    h->d = 0;
}

#ifdef __cplusplus
}
#endif
