/*
 * Copyright (c) 2013, The University of Oxford
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


#include <oskar_global.h>

#include <utility/oskar_get_error_string.h>
#include <utility/oskar_log_error.h>
#include <utility/oskar_mem_get_pointer.h>

#include <imaging/oskar_Image.h>
#include <imaging/oskar_image_read.h>

#include <apps/lib/oskar_OptionParser.h>

#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <algorithm>
#define PNG_DEBUG 3
#include <png.h>
#include <cfloat>


void abort_(const char* s, ...)
{
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

inline void setRGB(png_byte *ptr, float val)
{
    int v = (int)(val * 768);
    if (v < 0) v = 0;
    if (v > 768) v = 768;
    int offset = v % 256;

    if (v<256) {
        ptr[0] = 0;
        ptr[1] = 0;
        ptr[2] = offset;
    }
    else if (v<512) {
        ptr[0] = 0;
        ptr[1] = offset;
        ptr[2] = 255-offset;
    }
    else {
        ptr[0] = offset;
        ptr[1] = 255-offset;
        ptr[2] = 0;
    }
}

inline void setRGB2(png_byte *ptr, float val, float red_val, float blue_val)
{
    int v = (int)(1023 * (val - red_val) / (blue_val - red_val));
    if (v < 256)
    {
        ptr[0] = 255;
        ptr[1] = v;
        ptr[2] = 0;
    }
    else if (v < 512)
    {
        v -= 256;
        ptr[0] = 255-v;
        ptr[1] = 255;
        ptr[2] = 0;
    }
    else if (v < 768)
    {
        v -= 512;
        ptr[0] = 0;
        ptr[1] = 255;
        ptr[2] = v;
    }
    else
    {
        v -= 768;
        ptr[0] = 0;
        ptr[1] = 255-v;
        ptr[2] = 255;
    }
}


// based on example found at: http://zarb.org/~gc/html/libpng.html
int main(int argc, char** argv)
{
    int status = OSKAR_SUCCESS;

    oskar_OptionParser opt("oskar_image_to_png");
    opt.addRequired("OSKAR image");
    if (!opt.check_options(argc, argv)) return OSKAR_FAIL;

    // Load OSKAR image
    oskar_Image img;
    oskar_image_read(&img, opt.getArg(), 0, &status);
    if (status)
    {
        oskar_log_error(0, oskar_get_error_string(status));
        return status;
    }

    printf("Number of times: %i\n", img.num_times);
    printf("Number of channels: %i\n", img.num_channels);
    printf("Number of polarisations: %i\n", img.num_pols);
    printf("Number of pixels: %i x %i = %i\n", img.width, img.height,
            img.width * img.height);

    int num_pixels = img.width*img.height;
    // Get a single slice of the image
    oskar_Mem img_slice;
    int t = 0;
    int p = 0;
    int c = 0;
    printf("Creating a PNG of slice (Chan.Time.Pol): %i.%i.%i\n",c,t,p);
    int slice_offset = ((c * img.num_times + t) * img.num_pols + p) * num_pixels;
    oskar_mem_get_pointer(&img_slice, &(img.data), slice_offset,
            num_pixels, &status);

    int x, y;
    //int width, height;
    //png_byte color_type;
    png_byte bit_depth;
    png_structp png_ptr;
    png_infop info_ptr;
    //int number_of_passes;
    //png_bytep* row_pointers;

    // Create the file.
    const char* filename = "TEMP.png";
    FILE* fp = fopen(filename, "wb");
    if (!fp)
        abort_("Failed to open PNG file for writing");


    // Init.
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        abort_("png_create_write_struct() failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!png_ptr)
        abort_("png_create_info_struct() failed");

    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("Error during init_io");

    png_init_io(png_ptr, fp);

    // Write header
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("Error during write header");

    bit_depth = 8;
    png_set_IHDR(png_ptr, info_ptr,
            img.width, img.height, bit_depth,
            PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE,
            PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    // Write bytes
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("error during writing bytes");

    // Allocate memory for one row, 3 bytes per pixel - RGB.
    png_bytep row;
    //row = (png_bytep) malloc(3 * img.width * sizeof(png_byte));
    row = (png_bytep) malloc(3 * img.width * sizeof(png_byte));

    float* img_data = (float*)img_slice.data;

    //short fltInt16;
    //int fltInt32;
    float red = -FLT_MAX; // max
    float blue = FLT_MAX; // min
    for (int i = 0; i < num_pixels; ++i)
    {
        blue = std::min(blue, img_data[i]);
        red = std::max(red, img_data[i]);
    }

    printf("Image min = %f, max = %f\n", blue, red);

    for (y = 0; y < img.height; y++)
    {
        for (x = 0; x < img.width; x++)
        {
            //int idx = (img.height-y-1)*img.width + (img.width-x-1);
            int idx = (img.height-y-1)*img.width + x;
            setRGB2(&row[x*3], img_data[idx], red, blue);

//            float flt = img_data[y*img.width + x];
//            memcpy(&fltInt32, &flt, sizeof(float));
//            fltInt16 = (fltInt32 & 0x00FFFFFF) >> 14;
//            fltInt16 |= ((fltInt32 & 0x7f000000) >> 26) << 10;
//            fltInt16 |= ((fltInt32 & 0x80000000) >> 16);
//            row[x] = fltInt16;
        }
        png_write_row(png_ptr, row);
    }

    //png_write_image(png_ptr, row_pointers);

    // End write
    if (setjmp(png_jmpbuf(png_ptr)))
        abort_("error during end of write");

    png_write_end(png_ptr, NULL);

    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);

    return status;
}
