/*
 * Copyright (c) 2012, The University of Oxford
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


#include "oskar_global.h"
#include "imaging/oskar_Image.h"
#include "imaging/oskar_image_read.h"
#include "imaging/oskar_get_image_type_string.h"

#include "utility/oskar_get_error_string.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_realloc.h"

#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_BinaryHeader.h"
#include "utility/oskar_mem_binary_stream_read.h"
#include "utility/oskar_binary_tag_index_query.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_free.h"

#include "utility/oskar_get_data_type_string.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv)
{
    int error = OSKAR_SUCCESS;

    if (argc < 2)
    {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  oskar_image_summary [OSKAR image] <display log>\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Arguments:\n");
        fprintf(stderr, "  OSKAR image (required): Path of an OSKAR image file.\n");
        fprintf(stderr, "  display log (optional, default: 'f'): display the simulation log associated\n");
        fprintf(stderr, "        with the image, 't' or 'f'\n");

        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    const char* filename = argv[1];
    bool verbose = false;
    if (argc == 3)
    {
        if (strcmp(argv[2], "t") == 0)
        {
            verbose = true;
        }
        else if (strcmp(argv[2], "f") == 0)
        {
            verbose = false;
        }
        else
        {
            fprintf(stderr, "ERROR: Unrecognised value for argument 2, "
                    "allowed values are 't' or 'f'\n");
            return OSKAR_ERR_INVALID_ARGUMENT;
        }
    }
    try
    {
        // Load the image into memory.
        oskar_Image image;
        error = oskar_image_read(&image, filename, 0);
        if (error)
        {
            fprintf(stderr, "ERROR: Failed to open specified image file: %s.\n",
                    oskar_get_error_string(error));
            return error;
        }

        printf("\n");
        printf("- Image type .............. %s\n",
                oskar_get_image_type_string(image.image_type));
        printf("\n");
        printf("- Data type ............... %s\n",
                oskar_get_data_type_string(image.data.type));
        printf("\n");
        printf("- Field of view (degrees) . %f\n", image.fov_dec_deg);
        printf("\n");
        printf("- Image cube dimensions:\n");
        printf("  - Order ................. ");
        for (int i = 0; i < 5; ++i)
        {
            switch (image.dimension_order[i])
            {
                case OSKAR_IMAGE_DIM_RA:
                    printf("%s", "RA");
                    break;
                case OSKAR_IMAGE_DIM_DEC:
                    printf("%s", "Dec");
                    break;
                case OSKAR_IMAGE_DIM_POL:
                    printf("%s", "Pol");
                    break;
                case OSKAR_IMAGE_DIM_TIME:
                    printf("%s", "Time");
                    break;
                case OSKAR_IMAGE_DIM_CHANNEL:
                    printf("%s", "Channel");
                    break;
                default:
                    fprintf(stderr, "\nERROR: Unrecognised dimension ID.\n");
                    return OSKAR_FAIL;
            };
            if (i < 4) printf(", ");
        }
        printf("\n");
        printf("  - Size RA, Dec. (pixels)  %i x %i\n", image.width, image.height);
        printf("  - No. of polarisations .. %i\n", image.num_pols);
        printf("  - No. of times .......... %i\n", image.num_times);
        printf("  - No. of channels ....... %i\n", image.num_channels);
        printf("\n");
        printf("- Pointing centre:\n");
        printf("  - RA (degrees) .......... %f\n", image.centre_ra_deg);
        printf("  - Dec. (degrees) ........ %f\n", image.centre_dec_deg);
        printf("\n");
        printf("- Start time MJD UTC ...... %f\n", image.time_start_mjd_utc);
        printf("- Time increment (seconds)  %f\n", image.time_inc_sec);
        printf("\n");
        printf("- Start frequency (Hz) .... %e\n", image.freq_start_hz);
        printf("- Frequency increment (Hz)  %e\n", image.freq_inc_hz);
        printf("\n");

        // If verbose, print the run log.
        if (verbose)
        {
            oskar_BinaryTagIndex* index = NULL;
            FILE* stream = fopen(filename, "rb");
            if (!stream)
                return OSKAR_ERR_FILE_IO;
            oskar_binary_tag_index_create(&index, stream);
            size_t data_size = 0;
            long int data_offset = 0;

            if (oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                    OSKAR_TAG_RUN_LOG, 0, &data_size, &data_offset) == OSKAR_SUCCESS)
            {
                oskar_Mem temp(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
                error = oskar_mem_binary_stream_read(&temp, stream, &index,
                        OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0);
                if (error) return OSKAR_FAIL;
                oskar_mem_realloc(&temp, temp.num_elements + 1, &error);
                if (error) return OSKAR_FAIL;
                ((char*)temp.data)[temp.num_elements - 1] = 0;
                printf("- Run Log:\n");
                printf("%s", (char*)(temp.data));
            }
            fclose(stream);
            oskar_binary_tag_index_free(&index);
        }
    }
    catch (int code)
    {
        error = code;
    }

    // Check for errors.
    if (error)
    {
        fprintf(stderr, "ERROR: Run failed with code %i: %s.\n", error,
                oskar_get_error_string(error));
    }

    return error;
}
