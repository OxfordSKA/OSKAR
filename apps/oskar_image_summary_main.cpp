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


#include <oskar_Image.h>
#include <oskar_image_read.h>
#include <oskar_get_image_type_string.h>

#include <oskar_get_data_type_string.h>
#include <oskar_get_error_string.h>
#include <oskar_mem.h>
#include <oskar_version_string.h>

#include <oskar_BinaryTag.h>
#include <oskar_BinaryHeader.h>
#include <oskar_mem_binary_stream_read.h>
#include <oskar_binary_tag_index_query.h>
#include <oskar_binary_tag_index_free.h>
#include <oskar_binary_tag_index_query.h>
#include <oskar_binary_tag_index_create.h>
#include <oskar_binary_stream_read_oskar_version.h>

#include <apps/lib/oskar_OptionParser.h>

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv)
{
    int status = OSKAR_SUCCESS;

    oskar_OptionParser opt("oskar_image_summary", oskar_version_string());
    opt.addRequired("OSKAR image file");
    opt.addFlag("-l", "Display the simulation log associated with the image.",
            false, "--log");
    opt.addFlag("-s", "Display the image settings.", false, "--settings");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    const char* filename = opt.getArg(0);
    bool displayLog = opt.isSet("-l") ? true : false;
    bool displaySettings = opt.isSet("-s") ? true : false;

    // Load the image into memory.
    int vMajor, vMinor, vPatch;
    oskar_Image image;
    oskar_image_read(&image, filename, 0, &status);

    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        status = OSKAR_ERR_FILE_IO;
        fprintf(stderr, "ERROR: Failed to open specified image file: %s.\n",
                oskar_get_error_string(status));
        return status;
    }

    oskar_binary_stream_read_oskar_version(file, &vMajor, &vMinor, &vPatch,
            &status);
    // True if version 2.3 or older
    bool v232 = (vMajor <= 2 && vMinor <= 3 && vPatch <= 3) ? true : false;

    fclose(file);

    if (!(displayLog || displaySettings))
    {
        printf("\n");
        printf("- OSKAR version ........... %i.%i.%i\n", vMajor, vMinor, vPatch);

        printf("\n");
        printf("- Image type .............. %s\n",
                oskar_get_image_type_string(image.image_type));
        printf("\n");
        printf("- Data type ............... %s\n",
                oskar_get_data_type_string(oskar_mem_type(&image.data)));
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

        if (v232 == false)
        {
            printf("\n");
            printf("- Grid type ............... ");
            if (image.grid_type == OSKAR_IMAGE_GRID_TYPE_RECTILINEAR)
                printf("%s", "Rectilinear\n");
            else if (image.grid_type == OSKAR_IMAGE_GRID_TYPE_HEALPIX)
                printf("%s", "HEALPIX\n");
            else
                printf("%s", "Undefined\n");
            printf("\n");
            printf("- Coordinate frame ........ ");
            if (image.coord_frame == OSKAR_IMAGE_COORD_FRAME_EQUATORIAL)
                printf("%s", "Equatorial\n");
            else if (image.coord_frame == OSKAR_IMAGE_COORD_FRAME_HORIZON)
                printf("%s", "Horizon\n");
            else
                printf("%s", "Undefined\n");
        }

        if (v232 || image.grid_type == OSKAR_IMAGE_GRID_TYPE_RECTILINEAR)
        {
            printf("\n");
            printf("- Field of view (degrees) . %f\n", image.fov_dec_deg);
            if (v232 || image.grid_type == OSKAR_IMAGE_COORD_FRAME_EQUATORIAL)
            {
                printf("\n");
                printf("- Pointing centre:\n");
                printf("  - RA (degrees) .......... %f\n", image.centre_ra_deg);
                printf("  - Dec. (degrees) ........ %f\n", image.centre_dec_deg);
            }
        }

        printf("\n");
        printf("- Start time MJD UTC ...... %f\n", image.time_start_mjd_utc);
        printf("- Time increment (seconds)  %f\n", image.time_inc_sec);
        printf("\n");
        printf("- Start frequency (Hz) .... %e\n", image.freq_start_hz);
        printf("- Frequency increment (Hz)  %e\n", image.freq_inc_hz);
        printf("\n");

    }
    // If verbose, print the run log.
    if (displayLog)
    {
        oskar_BinaryTagIndex* index = NULL;
        FILE* stream = fopen(filename, "rb");
        if (!stream)
            return OSKAR_ERR_FILE_IO;
        oskar_binary_tag_index_create(&index, stream, &status);
        size_t data_size = 0;
        long int data_offset = 0;
        int tag_error = 0;

        oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                OSKAR_TAG_RUN_LOG, 0, &data_size, &data_offset, &tag_error);
        if (!tag_error)
        {
            oskar_Mem *temp;
            temp = oskar_mem_create(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, &status);
            oskar_mem_binary_stream_read(temp, stream, &index,
                    OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, &status);
            oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
            if (!status)
            {
                oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
                printf("%s", oskar_mem_char(temp));
            }
            oskar_mem_free(temp, &status);
            free(temp); // FIXME Remove after updating oskar_mem_free().
        }
        fclose(stream);
        oskar_binary_tag_index_free(index, &status);
    }

    if (displaySettings)
    {
        oskar_BinaryTagIndex* index = NULL;
        FILE* stream = fopen(filename, "rb");
        if (!stream)
            return OSKAR_ERR_FILE_IO;
        oskar_binary_tag_index_create(&index, stream, &status);
        size_t data_size = 0;
        long int data_offset = 0;
        int tag_error = 0;
        oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_SETTINGS,
                OSKAR_TAG_SETTINGS, 0, &data_size, &data_offset, &tag_error);
        if (!tag_error)
        {
            oskar_Mem *temp;
            temp = oskar_mem_create(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, &status);
            oskar_mem_binary_stream_read(temp, stream, &index,
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &status);
            oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
            if (!status)
            {
                oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
                printf("%s", oskar_mem_char(temp));
            }
            oskar_mem_free(temp, &status);
            free(temp); // FIXME Remove after updating oskar_mem_free().
        }
        fclose(stream);
        oskar_binary_tag_index_free(index, &status);
    }

    // Check for errors.
    if (status)
    {
        fprintf(stderr, "ERROR: Run failed with code %i: %s.\n", status,
                oskar_get_error_string(status));
    }

    return status;
}
