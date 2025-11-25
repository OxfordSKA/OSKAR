/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"


TEST(Mem, fits_healpix)
{
    int status = 0;
    const int nside_in = 128;
    const char ordering_in = 'R';
    const char coordsys_in = 'G';
    const int num_elem = 12 * nside_in * nside_in;
    const char* file_name = "temp_test_healpix.fits";
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};

    // Write, read and verify data in single and double precision.
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Create test data.
        oskar_Mem* in = oskar_mem_create(
                types[i_type], OSKAR_CPU, num_elem, &status
        );
        oskar_Mem* zeros = oskar_mem_create(
                types[i_type], OSKAR_CPU, num_elem, &status
        );
        oskar_mem_clear_contents(zeros, &status);
        oskar_mem_random_uniform(in, 1, 2, 3, 4, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Write the data.
        oskar_mem_write_healpix_fits(
                in, file_name, 1, nside_in, ordering_in, coordsys_in, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Read the data.
        int nside_out = 0;
        char ordering_out = (char) 0;
        char coordsys_out = (char) 0;
        char* brightness_units = 0;
        oskar_Mem* out = oskar_mem_read_healpix_fits(
                file_name, 0, &nside_out, &ordering_out, &coordsys_out,
                &brightness_units, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(ordering_in, ordering_out);
        ASSERT_EQ(coordsys_in, coordsys_out);
        EXPECT_EQ(NULL, brightness_units);

        // Verify the data.
        EXPECT_EQ(1, oskar_mem_different(out, zeros, 0, &status));
        EXPECT_EQ(0, oskar_mem_different(out, in, 0, &status));

        // Clean up.
        oskar_mem_free(in, &status);
        oskar_mem_free(out, &status);
        oskar_mem_free(zeros, &status);
        free(brightness_units);
        (void) remove(file_name);
    }
}


TEST(Mem, fits_healpix_not_a_fits_file)
{
    int status = 0;
    const char* file_name = "temp_test_healpix.data";

    // Write a non-FITS file.
    FILE* file = fopen(file_name, "w");
    (void) fprintf(file, "Hello\n");
    (void) fclose(file);

    // Attempt to read the data.
    // Expect failure.
    int nside_out = 0;
    char ordering_out = (char) 0;
    char coordsys_out = (char) 0;
    char* brightness_units = 0;
    oskar_Mem* out = oskar_mem_read_healpix_fits(
            file_name, 0, &nside_out, &ordering_out, &coordsys_out,
            &brightness_units, &status
    );
    ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
    ASSERT_EQ(NULL, out);

    // Clean up.
    (void) remove(file_name);
}


TEST(Mem, fits_healpix_not_a_healpix_fits_file)
{
    int status = 0;
    const char* file_name = "temp_test_healpix_not_a_healpix_file.fits";

    // Create test data.
    const int num_elem = 10;
    oskar_Mem* data = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_random_uniform(data, 1, 2, 3, 4, &status);

    // Write a FITS binary table that is not a HEALPix FITS file.
    oskar_mem_write_fits_bintable(
            file_name, "NOT_HEALPIX", 1, num_elem, &status, data
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Attempt to read the data.
    // Expect failure.
    int nside_out = 0;
    char ordering_out = (char) 0;
    char coordsys_out = (char) 0;
    char* brightness_units = 0;
    oskar_Mem* out = oskar_mem_read_healpix_fits(
            file_name, 0, &nside_out, &ordering_out, &coordsys_out,
            &brightness_units, &status
    );
    ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
    ASSERT_EQ(NULL, out);

    // Clean up.
    oskar_mem_free(data, &status);
    (void) remove(file_name);
}


TEST(Mem, fits_healpix_wrong_hdu_index)
{
    int status = 0;
    const int nside_in = 128;
    const char ordering_in = 'R';
    const char coordsys_in = 'G';
    const int num_elem = 12 * nside_in * nside_in;
    const char* file_name = "temp_test_healpix_wrong_hdu_index.fits";

    // Create test data.
    oskar_Mem* data = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_random_uniform(data, 1, 2, 3, 4, &status);

    // Write a HEALPix FITS file.
    oskar_mem_write_healpix_fits(
            data, file_name, 1,
            nside_in, ordering_in, coordsys_in, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Attempt to read the data and request a HDU index that is out of range.
    // Expect failure.
    int nside_out = 0;
    char ordering_out = (char) 0;
    char coordsys_out = (char) 0;
    char* brightness_units = 0;
    oskar_Mem* out = oskar_mem_read_healpix_fits(
            file_name, 10, &nside_out, &ordering_out, &coordsys_out,
            &brightness_units, &status
    );
    ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
    ASSERT_EQ(NULL, out);

    // Clean up.
    oskar_mem_free(data, &status);
    (void) remove(file_name);
}
