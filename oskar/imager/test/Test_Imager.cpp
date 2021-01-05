/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "imager/oskar_imager.h"
#include "vis/oskar_vis_header.h"
#include "vis/oskar_vis_block.h"

#define WRITE_FITS 1

TEST(imager, update_from_block)
{
    int status = 0, type = OSKAR_DOUBLE;
    const int size = 1024, grid_size = size * size;

    // Create and set up an imager.
    oskar_Imager* im = oskar_imager_create(type, &status);
    ASSERT_EQ(0, status);
    oskar_imager_set_fov(im, 5.0);
    oskar_imager_set_size(im, size, &status);
    oskar_imager_set_weighting(im, "Uniform", &status);
    ASSERT_EQ(0, status);

    // Test accessors.
    ASSERT_DOUBLE_EQ(0.0, oskar_imager_freq_min_hz(im));
    ASSERT_DOUBLE_EQ(0.0, oskar_imager_freq_max_hz(im));
    ASSERT_DOUBLE_EQ(0.0, oskar_imager_time_min_utc(im));
    ASSERT_DOUBLE_EQ(0.0, oskar_imager_time_max_utc(im));
    ASSERT_DOUBLE_EQ(0.0, oskar_imager_uv_filter_min(im));
    ASSERT_DOUBLE_EQ(DBL_MAX, oskar_imager_uv_filter_max(im));
    ASSERT_EQ(type, oskar_imager_precision(im));
    ASSERT_EQ(size, oskar_imager_size(im));
    ASSERT_STREQ("I", oskar_imager_image_type(im));
    ASSERT_STREQ("Uniform", oskar_imager_weighting(im));

    // Create visibility data.
    const int num_times = 8, num_channels = 1, num_stations = 64;
    oskar_VisHeader* hdr = oskar_vis_header_create(type | OSKAR_COMPLEX, type,
            num_times, num_times, num_channels, num_channels,
            num_stations, 0, 1, &status);
    ASSERT_EQ(0, status);
    oskar_vis_header_set_freq_start_hz(hdr, 100e6);
    oskar_VisBlock* block = oskar_vis_block_create_from_header(
            OSKAR_CPU, hdr, &status);
    ASSERT_EQ(0, status);
    oskar_Mem* vis = oskar_vis_block_cross_correlations(block);
    oskar_Mem* u = oskar_vis_block_station_uvw_metres(block, 0);
    oskar_Mem* v = oskar_vis_block_station_uvw_metres(block, 1);
    oskar_Mem* w = oskar_vis_block_station_uvw_metres(block, 2);
    ASSERT_EQ(0, status);
    oskar_mem_random_gaussian(u, 0, 1, 2, 3, 500.0, &status);
    oskar_mem_random_gaussian(v, 4, 5, 6, 7, 500.0, &status);
    oskar_mem_set_value_real(w, 0.0, 0, oskar_mem_length(w), &status);
    for (size_t i = 0; i < oskar_mem_length(vis); ++i)
    {
        oskar_mem_double2(vis, &status)[i].x = 1.0;
        oskar_mem_double2(vis, &status)[i].y = 4.0;
    }
    ASSERT_EQ(0, status);

    // Process visibility data.
    oskar_imager_set_coords_only(im, 1);
    ASSERT_EQ(1, oskar_imager_coords_only(im));
    oskar_imager_update_from_block(im, hdr, block, &status);
    oskar_imager_set_coords_only(im, 0);
    oskar_imager_check_init(im, &status);
    ASSERT_EQ(1, oskar_imager_num_image_planes(im));
    ASSERT_EQ(0, oskar_imager_coords_only(im));
    oskar_imager_update_from_block(im, hdr, block, &status);
    ASSERT_EQ(0, status);

    // Finalise the image.
    oskar_Mem* image = oskar_mem_create(type, OSKAR_CPU, grid_size, &status);
    oskar_Mem* grid = oskar_mem_create(type | OSKAR_COMPLEX, OSKAR_CPU,
            grid_size, &status);
    oskar_imager_finalise(im, 1, &image, 1, &grid, &status);
    ASSERT_EQ(0, status);

#ifdef WRITE_FITS
    // Save the image and the grid.
    oskar_mem_write_fits_cube(image, "test_imager_update_from_block_image.fits",
            size, size, 1, 0, &status);
    oskar_mem_write_fits_cube(grid, "test_imager_update_from_block_grid.fits",
            size, size, 1, 0, &status);
#endif

    // Clean up.
    oskar_imager_free(im, &status);
    oskar_vis_block_free(block, &status);
    oskar_vis_header_free(hdr, &status);
    oskar_mem_free(image, &status);
    oskar_mem_free(grid, &status);
}
