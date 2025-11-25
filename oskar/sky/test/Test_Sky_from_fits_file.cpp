/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>

#include <fitsio.h>
#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_get_error_string.h"

#define DEG2RAD (M_PI / 180.0)


static void write_test_fits_healpix(
        int type,
        int nside,
        const char* filename,
        int* status
)
{
    const int num_pixels = 12 * nside * nside;
    oskar_Mem* in = oskar_mem_create(type, OSKAR_CPU, num_pixels, status);
    for (int i = 0; i < num_pixels; ++i)
    {
        oskar_mem_set_element_real(in, i, (double) i + 1., status);
    }
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
    oskar_mem_write_healpix_fits(in, filename, true, nside, 'R', 'E', status);
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
    oskar_mem_free(in, status);
}


static void write_test_fits_image(int type, const char* filename, int* status)
{
    fitsfile *fptr;
    long naxes[2] = {256, 256};
    long fpixel[2] = {1, 1};

    /* Allocate an image and insert some non-zero pixels
     * so there's something to load. */
    /* Do not change - hardcoded asserts! */
    oskar_Mem* data = oskar_mem_create(
            type, OSKAR_CPU, naxes[0] * naxes[1], status
    );
    oskar_mem_set_element_real(data,  64 * naxes[0] + 200, 5.123, status);
    oskar_mem_set_element_real(data, 128 * naxes[0] + 128, 10.24, status);
    oskar_mem_set_element_real(data, 200 * naxes[0] +  40, 3.456, status);
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);

    /* Create FITS file */
    if (oskar_file_exists(filename)) (void) remove(filename);
    fits_create_file(&fptr, filename, status);
    fits_create_img(
            fptr, type == OSKAR_SINGLE ? FLOAT_IMG : DOUBLE_IMG,
            2, naxes, status
    );
    fits_write_key_str(fptr, "CTYPE1", "RA---SIN",  NULL, status);
    fits_write_key_str(fptr, "CTYPE2", "DEC--SIN",  NULL, status);

    /* Simple WCS header. */
    /* Do not change - hardcoded asserts! */
    const int decimals = 15;
    double crval1 = 180.0;
    double crval2 =  45.0;
    double crpix1 = 129;
    double crpix2 = 129;
    double cdelt1 = -0.00027778; /* 1 arcsecond/pixel. */
    double cdelt2 =  0.00027778;
    fits_write_key_dbl(fptr, "CRVAL1", crval1, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRVAL2", crval2, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRPIX1", crpix1, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CRPIX2", crpix2, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CDELT1", cdelt1, decimals, NULL, status);
    fits_write_key_dbl(fptr, "CDELT2", cdelt2, decimals, NULL, status);
    fits_write_key_str(fptr, "BUNIT", "Jy/pixel", "Brightness units", status);
    fits_write_key_dbl(fptr, "EQUINOX", 2000, 1, NULL, status);

    /* Write image. */
    fits_write_pix(
            fptr, type == OSKAR_SINGLE ? TFLOAT : TDOUBLE,
            fpixel, naxes[0] * naxes[1], oskar_mem_void(data), status
    );

    /* Clean up. */
    fits_close_file(fptr, status);
    if (status) fits_report_error(stderr, *status);
    oskar_mem_free(data, status);
}


TEST(Sky, from_fits_healpix)
{
    const int nside = 2;
    const int num_pixels = 12 * nside * nside;
    const char* filename = "temp_test_load_fits_healpix.fits";
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerance[] = {1e-6, 1e-12};

    // Load a FITS file in both single and double precision.
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Write out a test HEALPix FITS file.
        int status = 0;
        write_test_fits_healpix(types[i_type], nside, filename, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the HEALPix FITS file.
        oskar_Sky* sky = oskar_sky_from_fits_file(
                types[i_type], filename,
                0.0, 0.0, "Jy/pixel", 0, 100e6, -0.7, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that things are as expected.
        const double tol = tolerance[i_type];
        ASSERT_EQ(num_pixels, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        for (int i = 0; i < num_pixels; ++i)
        {
            ASSERT_NEAR(
                    (double) i + 1.,
                    oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, i), tol
            );
        }

        // Clean up.
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
}


TEST(Sky, from_fits_image_file)
{
    const char* filename = "temp_test_load_fits_image.fits";
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerance[] = {1e-6, 1e-12};

    // Load a FITS file in both single and double precision.
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Write out a test FITS file.
        int status = 0;
        write_test_fits_image(types[i_type], filename, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Load the FITS file.
        oskar_Sky* sky = oskar_sky_from_fits_file(
                types[i_type], filename,
                0.0, 0.0, "Jy/pixel", 0, 100e6, -0.7, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that things are as expected.
        const double tol = tolerance[i_type];
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_NEAR(5.123, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0), tol);
        ASSERT_NEAR(10.24, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1), tol);
        ASSERT_NEAR(3.456, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 2), tol);
        ASSERT_NEAR(
                179.971724 * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0), 1e-6
        );
        ASSERT_NEAR(
                180.0 * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1), tol
        );
        ASSERT_NEAR(
                180.034582 * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 2), 1e-6
        );
        ASSERT_NEAR(
                44.9822186 * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0), 1e-6
        );
        ASSERT_NEAR(
                45.0 * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1), tol
        );
        ASSERT_NEAR(
                45.019995 * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 2), 1e-6
        );

        // Save sky model to text file for checking.
        // oskar_sky_save_named_columns(
        //         sky, (std::string(filename) + ".txt").c_str(),
        //         false, true, true, &status
        // );

        // Clean up.
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
}
