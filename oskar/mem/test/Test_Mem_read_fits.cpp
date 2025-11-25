/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>

#include <fitsio.h>
#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_get_error_string.h"


static void write_test_fits_cube(int type, const char* filename, int* status)
{
    fitsfile *fptr;
    long naxes[3] = {256, 256, 4};
    long fpixel[3] = {1, 1, 1};

    /* Allocate a cube and fill it with a pattern. */
    oskar_Mem* data = oskar_mem_create(
            type, OSKAR_CPU, naxes[0] * naxes[1] * naxes[2], status
    );
    for (int k = 0, p = 0; k < naxes[2]; ++k)
    {
        for (int j = 0; j < naxes[1]; ++j)
        {
            for (int i = 0; i < naxes[0]; ++i, ++p)
            {
                oskar_mem_set_element_real(data, p, (double) p, status);
            }
        }
    }
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);

    /* Create FITS file */
    if (oskar_file_exists(filename)) (void) remove(filename);
    fits_create_file(&fptr, filename, status);
    fits_create_img(
            fptr, type == OSKAR_SINGLE ? FLOAT_IMG : DOUBLE_IMG,
            3, naxes, status
    );
    fits_write_key_str(fptr, "CTYPE1", "RA---SIN",  NULL, status);
    fits_write_key_str(fptr, "CTYPE2", "DEC--SIN",  NULL, status);
    fits_write_key_str(fptr, "CTYPE3", "FREQ",  NULL, status);

    /* Simple WCS header. */
    const int decimals = 15;
    double crval1 = 180.0;
    double crval2 =  45.0;
    double crval3 = 100e6;
    double crpix1 = 129;
    double crpix2 = 129;
    double crpix3 = 1;
    double cdelt1 = -0.00027778; /* 1 arcsecond/pixel. */
    double cdelt2 =  0.00027778;
    double cdelt3 =  1e6;
    fits_write_key_dbl(fptr, "CRVAL1", crval1, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRVAL2", crval2, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRVAL3", crval3, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRPIX1", crpix1, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRPIX2", crpix2, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRPIX3", crpix3, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CDELT1", cdelt1, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CDELT2", cdelt2, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CDELT3", cdelt3, decimals, NULL, status);
    fits_write_key_str(fptr, "BUNIT", "Jy/pixel", "Brightness units", status);
    fits_write_key_dbl(fptr, "EQUINOX", 2000, 1, NULL, status);

    /* Write image. */
    fits_write_pix(
            fptr, type == OSKAR_SINGLE ? TFLOAT : TDOUBLE,
            fpixel, naxes[0] * naxes[1] * naxes[2],
            oskar_mem_void(data), status
    );

    /* Clean up. */
    fits_close_file(fptr, status);
    if (status) fits_report_error(stderr, *status);
    oskar_mem_free(data, status);
}


TEST(Mem, read_fits)
{
    int status = 0;
    const char* file_name = "temp_test_mem_read.fits";
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerances[] = {5e-5, 1e-14};

    // Write, read and verify data in single and double precision.
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Write the data.
        const int image_size = 256;
        write_test_fits_cube(types[i_type], file_name, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create an empty array to fill.
        oskar_Mem* data = oskar_mem_create(
                types[i_type], OSKAR_CPU, image_size * image_size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Read a single image plane.
        const int i_plane = 2;
        int num_axes = 0;
        int* axis_size = 0;
        double* axis_inc = 0;
        const int num_index_dims = 3;
        const int start_index[] = {0, 0, i_plane};
        oskar_mem_read_fits(
                data, 0, image_size * image_size, file_name,
                num_index_dims, start_index,
                &num_axes, &axis_size, &axis_inc, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, num_axes);
        ASSERT_EQ(image_size, axis_size[0]);
        ASSERT_EQ(image_size, axis_size[1]);
        ASSERT_EQ(4, axis_size[2]);

        // Verify the data.
        const double tol = tolerances[i_type];
        for (int j = 0, p = 0; j < image_size; ++j)
        {
            for (int i = 0; i < image_size; ++i, ++p)
            {
                double expected = (double) (
                        i_plane * image_size * image_size + j * image_size + i
                );
                ASSERT_NEAR(
                        expected, oskar_mem_get_element(data, p, &status), tol
                );
            }
        }

        // Clean up.
        oskar_mem_free(data, &status);
        free(axis_size);
        free(axis_inc);
        (void) remove(file_name);
    }
}


TEST(Mem, read_fits_not_a_fits_file)
{
    int status = 0;
    const char* file_name = "temp_test_read_fits.data";

    // Write a non-FITS file.
    FILE* file = fopen(file_name, "w");
    (void) fprintf(file, "Hello\n");
    (void) fclose(file);

    // Attempt to read the data.
    const int num_pixels = 10;
    oskar_Mem* data = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_pixels, &status
    );

    // Expect failure.
    int num_axes = 0;
    int* axis_size = 0;
    double* axis_inc = 0;
    const int num_index_dims = 1;
    const int start_index[] = {0};
    oskar_mem_read_fits(
            data, 0, num_pixels, file_name, num_index_dims, start_index,
            &num_axes, &axis_size, &axis_inc, &status
    );
    ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
    ASSERT_EQ(NULL, axis_size);
    ASSERT_EQ(NULL, axis_inc);

    // Clean up.
    oskar_mem_free(data, &status);
    (void) remove(file_name);
}
