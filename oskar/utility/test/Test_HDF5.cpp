/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_hdf5.h"
#include <cstdlib>

#define INDEX_4D(N4, N3, N2, N1, I4, I3, I2, I1) \
    (N1 * (N2 * (N3 * I4 + I3) + I2) + I1)


static void print_attributes(
        int num_attributes,
        oskar_Mem** names,
        oskar_Mem** values,
        int* status
)
{
    for (int i = 0; i < num_attributes; ++i)
    {
        printf(
                "Attribute %d/%d: %s\n",
                i + 1, num_attributes, oskar_mem_char(names[i])
        );
        if (oskar_mem_type(values[i]) == OSKAR_CHAR)
        {
            printf("'%s'\n", oskar_mem_char(values[i]));
        }
        else
        {
            oskar_mem_save_ascii(
                    stdout, 1, 0, oskar_mem_length(values[i]), status,
                    values[i]
            );
        }
    }
}


TEST(HDF5, test_basic_read_write)
{
    int status = 0;
    const int num_elem = 10;

    // Create test data.
    oskar_Mem* int_in = oskar_mem_create(
            OSKAR_INT, OSKAR_CPU, 2 * num_elem, &status
    );
    oskar_Mem* float_in = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
    );
    oskar_Mem* double_in = oskar_mem_create(
            OSKAR_DOUBLE, OSKAR_CPU, 2 * num_elem, &status
    );
    oskar_Mem* c_float_in = oskar_mem_create(
            OSKAR_SINGLE_COMPLEX, OSKAR_CPU, num_elem, &status
    );
    oskar_Mem* c_double_in = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_random_uniform(float_in, 1, 2, 3, 4, &status);
    oskar_mem_random_uniform(double_in, 5, 6, 7, 8, &status);
    oskar_mem_random_uniform(c_float_in, 9, 10, 11, 12, &status);
    oskar_mem_random_uniform(c_double_in, 13, 14, 15, 16, &status);
    for (int i = 0; i < 2 * num_elem; ++i)
    {
        oskar_mem_int(int_in, &status)[i] = i;
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create the HDF5 file.
    const char* filename = "temp_test_hdf5_basic.h5";
    oskar_HDF5* h = oskar_hdf5_open(filename, 'a', &status);

    // Create the groups.
    oskar_hdf5_write_group(h, 0, "Grp1", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_hdf5_write_group(h, 0, "/Grp2", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write the data.
    oskar_hdf5_write_dataset(h, "", "int", 0, 0, int_in, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_hdf5_write_dataset(h, "/Grp1", "int", 0, 0, int_in, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_hdf5_write_dataset(h, "/Grp1", "float", 0, 0, float_in, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_hdf5_write_dataset(h, "Grp1", "double", 0, 0, double_in, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_hdf5_write_dataset(h, "", "c_float", 0, 0, c_float_in, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_hdf5_write_dataset(
            h, "Grp2", "c_double", 0, 0, c_double_in, 0, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try some invalid writes.
    {
        int status = 0;
        oskar_hdf5_write_dataset(h, 0, 0, 0, 0, c_double_in, 0, &status);
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    }
    {
        int status = 0;
        oskar_hdf5_write_dataset(h, "", "", 0, 0, c_double_in, 0, &status);
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    }

    // Read the data.
    oskar_Mem* int_out_root = oskar_hdf5_read_dataset(
            h, "", "int", 0, 0, &status
    );
    oskar_Mem* int_out = oskar_hdf5_read_dataset(
            h, "Grp1", "int", 0, 0, &status
    );
    oskar_Mem* float_out = oskar_hdf5_read_dataset(
            h, "/Grp1", "float", 0, 0, &status
    );
    oskar_Mem* double_out = oskar_hdf5_read_dataset(
            h, "Grp1", "double", 0, 0, &status
    );
    oskar_Mem* c_float_out = oskar_hdf5_read_dataset(
            h, 0, "c_float", 0, 0, &status
    );
    oskar_Mem* c_double_out = oskar_hdf5_read_dataset(
            h, "/Grp2", "c_double", 0, 0, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try some invalid reads.
    {
        int status = 0;
        oskar_Mem* err1 = oskar_hdf5_read_dataset(h, 0, 0, 0, 0, &status);
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(NULL, err1);
    }
    {
        int status = 0;
        oskar_Mem* err2 = oskar_hdf5_read_dataset(h, "", "", 0, 0, &status);
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(NULL, err2);
    }
    {
        int status = 0;
        int num_dims = 0;
        oskar_hdf5_read_dataset_dims(
                h, "/", "", &num_dims, 0, &status
        );
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    }

    // Verify the data.
#if 0
    // Printing disabled.
    printf("Int data:\n");
    oskar_mem_save_ascii(
            stdout, 2, 0, oskar_mem_length(int_in), &status, int_in, int_out
    );
    printf("Float data:\n");
    oskar_mem_save_ascii(
            stdout, 2, 0, oskar_mem_length(float_in), &status,
            float_in, float_out
    );
    printf("Double data:\n");
    oskar_mem_save_ascii(
            stdout, 2, 0, oskar_mem_length(double_in), &status,
            double_in, double_out
    );
    printf("Complex float data:\n");
    oskar_mem_save_ascii(
            stdout, 2, 0, oskar_mem_length(c_float_in), &status,
            c_float_in, c_float_out
    );
    printf("Complex double data:\n");
    oskar_mem_save_ascii(
            stdout, 2, 0, oskar_mem_length(c_double_in), &status,
            c_double_in, c_double_out
    );
#endif
    oskar_Mem* float_zeros = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, oskar_mem_length(float_out), &status
    );
    oskar_mem_clear_contents(float_zeros, &status);
    EXPECT_EQ(1, oskar_mem_different(float_out, float_zeros, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(int_out, int_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(int_out_root, int_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(float_out, float_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(double_out, double_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(c_float_out, c_float_in, 0, &status));
    EXPECT_EQ(0, oskar_mem_different(c_double_out, c_double_in, 0, &status));

    // Clean up.
    oskar_mem_free(int_in, &status);
    oskar_mem_free(float_in, &status);
    oskar_mem_free(double_in, &status);
    oskar_mem_free(c_float_in, &status);
    oskar_mem_free(c_double_in, &status);
    oskar_mem_free(float_zeros, &status);
    oskar_mem_free(int_out, &status);
    oskar_mem_free(int_out_root, &status);
    oskar_mem_free(float_out, &status);
    oskar_mem_free(double_out, &status);
    oskar_mem_free(c_float_out, &status);
    oskar_mem_free(c_double_out, &status);
    oskar_hdf5_close(h);
    remove(filename);
}


TEST(HDF5, test_vis_read_write)
{
    int status = 0;
    const int num_times = 8, num_baselines = 153, num_chan = 12;
    const int num_elem = num_times * num_baselines * num_chan;
    const char* filename = "temp_test_hdf5_vis.h5";

    // Create test data.
    oskar_Mem* in = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, num_elem, &status
    );
    oskar_mem_random_uniform(in, 1, 2, 3, 4, &status);

    // Create the HDF5 file.
    {
        int status = 0;
        oskar_HDF5* h = oskar_hdf5_open(filename, 'a', &status);

        // Write the data.
        size_t dims0[] = {num_times, num_baselines, num_chan};
        oskar_hdf5_write_dataset(h, "/", "vis", 3, dims0, in, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_close(h);
    }

    // Open the file for reading.
    {
        oskar_HDF5* h = oskar_hdf5_open(filename, 'r', &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(1, oskar_hdf5_dataset_exists(h, "/vis"));
        ASSERT_EQ(0, oskar_hdf5_dataset_exists(h, "/unknown_dataset"));

        // Check the data dimensions.
        int num_dims = 0;
        size_t* dims1 = 0;
        oskar_hdf5_read_dataset_dims(h, "", "vis", &num_dims, &dims1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(4, num_dims);
        ASSERT_EQ(num_times, (int) dims1[0]);
        ASSERT_EQ(num_baselines, (int) dims1[1]);
        ASSERT_EQ(num_chan, (int) dims1[2]);
        ASSERT_EQ(4, (int) dims1[3]);
        free(dims1);

        // Increment the reference count,
        // so we will need to close the file twice.
        oskar_hdf5_ref_inc(h);
        oskar_hdf5_close(h); // First close.

        // Read the data. The file should still be open.
        size_t* dims2 = 0;
        oskar_Mem* out = oskar_hdf5_read_dataset(
                h, 0, "vis", &num_dims, &dims2, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(4, num_dims);
        ASSERT_EQ(num_times, (int) dims2[0]);
        ASSERT_EQ(num_baselines, (int) dims2[1]);
        ASSERT_EQ(num_chan, (int) dims2[2]);
        ASSERT_EQ(4, (int) dims2[3]);
        free(dims2);

        // Verify the data.
        ASSERT_EQ(4 * num_elem, (int) oskar_mem_length(out));
        int num_to_check = 8 * num_elem;
        double* in_ = oskar_mem_double(in, &status);
        double* out_ = oskar_mem_double(out, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        for (int i = 0; i < num_to_check; ++i)
        {
            ASSERT_DOUBLE_EQ(in_[i], out_[i]);
        }

        // Clean up.
        oskar_mem_free(out, &status);
        oskar_hdf5_close(h); // Second close.
    }

    // Clean up.
    oskar_mem_free(in, &status);
    remove(filename);
}


TEST(HDF5, test_vis_block_read_write)
{
    int status = 0;

    // Block dimensions.
    const int num_time_block = 2;
    const int num_baselines_block = 789;
    const int num_chan_block = 4;
    const int num_elem = num_time_block * num_baselines_block * num_chan_block;

    // Number of blocks.
    const int num_blocks_time = 8;
    const int num_blocks_chan = 6;

    // Overall dimensions.
    const int num_time = num_time_block * num_blocks_time;
    const int num_baselines = num_baselines_block;
    const int num_chan = num_chan_block * num_blocks_chan;

    // Create the HDF5 file.
    const char* filename = "temp_test_hdf5_vis_block.h5";
    oskar_HDF5* h = oskar_hdf5_open(filename, 'w', &status);

    // Create a visibility block.
    oskar_Mem* block = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, num_elem, &status
    );
    double2* data = oskar_mem_double2(block, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write an empty dataset, big enough for the whole array.
    size_t dims0[] = {num_time, num_baselines, num_chan};
    oskar_hdf5_write_dataset(h, "/", "vis", 3, dims0, block, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try writing a hyperslab with invalid arguments.
    {
        int status = 0;
        oskar_hdf5_write_hyperslab(h, "/", "", 0, 0, 0, block, &status);
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    }

    // Loop over blocks.
    for (int ot = 0; ot < num_blocks_time; ++ot)
    {
        for (int oc = 0; oc < num_blocks_chan; ++oc)
        {
            // Fill the data block with a pattern for checking.
            oskar_mem_clear_contents(block, &status);
            for (int bt = 0; bt < num_time_block; ++bt)
            {
                for (int bb = 0; bb < num_baselines_block; ++bb)
                {
                    for (int bc = 0; bc < num_chan_block; ++bc)
                    {
                        for (int bp = 0; bp < 4; ++bp)
                        {
                            const int i_time = ot * num_time_block + bt;
                            const int i_chan = oc * num_chan_block + bc;
                            const double2 value = {
                                (double) (100000 * i_time + 1000 * i_chan + bb),
                                (double) bp
                            };
                            data[INDEX_4D(
                                    num_time_block,
                                    num_baselines_block,
                                    num_chan_block,
                                    4,
                                    bt,
                                    bb,
                                    bc,
                                    bp
                            )] = value;
                        }
                    }
                }
            }

            // Write the data block as a hyperslab.
            size_t offset[] = {
                (size_t) (ot * num_time_block),
                (size_t) 0,
                (size_t) (oc * num_chan_block)
            };
            size_t dims[] = {
                (size_t) num_time_block,
                (size_t) num_baselines_block,
                (size_t) num_chan_block
            };
            oskar_hdf5_write_hyperslab(
                    h, "/", "vis", 3, offset, dims, block, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
    }
    oskar_mem_free(block, &status);
    oskar_hdf5_close(h);

    // Open the HDF5 file to read the pattern back out.
    h = oskar_hdf5_open(filename, 'a', &status);

    // Loop over blocks.
    for (int ot = 0; ot < num_blocks_time; ++ot)
    {
        for (int oc = 0; oc < num_blocks_chan; ++oc)
        {
            // Read the data block as a hyperslab.
            size_t dims_block[] = {
                (size_t) num_time_block,
                (size_t) num_baselines_block,
                (size_t) num_chan_block,
                (size_t) 4
            };
            size_t offset[] = {
                (size_t) ot * num_time_block,
                (size_t) 0,
                (size_t) oc * num_chan_block,
                (size_t) 0
            };
            oskar_Mem* block = oskar_hdf5_read_hyperslab(
                    h, "/", "vis", 4, offset, dims_block, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Check the pattern in the data is correct.
            double2* data = oskar_mem_double2(block, &status);
            for (int bt = 0; bt < num_time_block; ++bt)
            {
                for (int bb = 0; bb < num_baselines_block; ++bb)
                {
                    for (int bc = 0; bc < num_chan_block; ++bc)
                    {
                        for (int bp = 0; bp < 4; ++bp)
                        {
                            const int i_time = ot * num_time_block + bt;
                            const int i_chan = oc * num_chan_block + bc;
                            const double2 expected_value = {
                                (double) (100000 * i_time + 1000 * i_chan + bb),
                                (double) bp
                            };
                            const double2 actual_value = data[INDEX_4D(
                                    num_time_block,
                                    num_baselines_block,
                                    num_chan_block,
                                    4,
                                    bt,
                                    bb,
                                    bc,
                                    bp
                            )];
                            ASSERT_DOUBLE_EQ(expected_value.x, actual_value.x);
                            ASSERT_DOUBLE_EQ(expected_value.y, actual_value.y);
                        }
                    }
                }
            }
            oskar_mem_free(block, &status);
        }
    }

    // Try reading a hyperslab with invalid arguments.
    {
        int status = 0;
        oskar_Mem* data = oskar_hdf5_read_hyperslab(
                h, 0, 0, 0, 0, 0, &status
        );
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(NULL, data);
    }
    {
        int status = 0;
        oskar_Mem* data = oskar_hdf5_read_hyperslab(
                h, "", "", 0, 0, 0, &status
        );
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(NULL, data);
    }
    {
        int status = 0;
        oskar_Mem* data = oskar_hdf5_read_hyperslab(
                h, "/", "vis", 1, 0, 0, &status
        );
        ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(NULL, data);
    }

    // Clean up.
    oskar_hdf5_close(h);
    remove(filename);
}


TEST(HDF5, test_attributes)
{
    // Write the HDF5 file.
    const char* filename = "temp_test_hdf5_attributes.h5";
    {
        int status = 0;

        // Create test data.
        const int num_elem = 10;
        oskar_Mem* data = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, num_elem, &status
        );

        // Create a HDF5 file.
        oskar_HDF5* h = oskar_hdf5_open(filename, 'w', &status);

        // Create the groups.
        oskar_hdf5_write_group(h, 0, "Group1", &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_group(h, "Group1", "Subgroup1", &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Write the data.
        oskar_hdf5_write_dataset(h, 0, "test", 0, 0, data, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Write the attributes.
        oskar_hdf5_write_attribute_int(h, 0, "int_attr1", 42, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_int(h, "/", "int_attr2", 99, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_double(h, "/", "dbl_attr1", 3.14, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_double(h, "/", "dbl_attr2", 2.718, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_double(
                h, "/test", "dataset_attr1", 2.99792458e8, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_double(
                h, "/test", "dataset_attr2", 6.63e-34, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_string(
                h, 0, "str_attr1",
                "A string attribute on the root group", &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_string(
                h, "Group1", "str_attr_group1",
                "A string attribute on Group1", &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_hdf5_write_attribute_string(
                h, "/Group1/Subgroup1", "str_attr_subgroup1",
                "A string attribute on Subgroup1", &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Clean up.
        oskar_mem_free(data, &status);
        oskar_hdf5_close(h);
    }

    // Read and verify the attributes one by one.
    {
        // Open the HDF5 file for reading.
        int status = 0;
        oskar_HDF5* h = oskar_hdf5_open(filename, 'r', &status);
        {
            int status = 0;
            int attribute = oskar_hdf5_read_attribute_int(
                    h, "/", "int_attr1", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_EQ(42, attribute);
        }
        {
            int status = 0;
            int attribute = oskar_hdf5_read_attribute_int(
                    h, 0, "int_attr2", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_EQ(99, attribute);
        }
        {
            int status = 0;
            double attribute = oskar_hdf5_read_attribute_double(
                    h, "/", "dbl_attr1", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_DOUBLE_EQ(3.14, attribute);
        }
        {
            int status = 0;
            double attribute = oskar_hdf5_read_attribute_double(
                    h, "/", "dbl_attr2", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_DOUBLE_EQ(2.718, attribute);
        }
        {
            int status = 0;
            double attribute = oskar_hdf5_read_attribute_double(
                    h, "test", "dataset_attr1", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_DOUBLE_EQ(2.99792458e8, attribute);
        }
        {
            int status = 0;
            double attribute = oskar_hdf5_read_attribute_double(
                    h, "/test", "dataset_attr2", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_DOUBLE_EQ(6.63e-34, attribute);
        }
        {
            int status = 0;
            char* attribute = oskar_hdf5_read_attribute_string(
                    h, "/", "str_attr1", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_STREQ("A string attribute on the root group", attribute);
            free(attribute);
        }
        {
            int status = 0;
            char* attribute = oskar_hdf5_read_attribute_string(
                    h, "Group1", "str_attr_group1", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_STREQ("A string attribute on Group1", attribute);
            free(attribute);
        }
        {
            int status = 0;
            char* attribute = oskar_hdf5_read_attribute_string(
                    h, "Group1/Subgroup1", "str_attr_subgroup1", &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            ASSERT_STREQ("A string attribute on Subgroup1", attribute);
            free(attribute);
        }

        // Read all the attributes on the dataset.
        int num_attr = 0;
        oskar_Mem** names = 0;
        oskar_Mem** values = 0;
        oskar_hdf5_read_attributes(
                h, "/test", &num_attr, &names, &values, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Verify the attributes.
        ASSERT_EQ(2, num_attr);
        ASSERT_STREQ("dataset_attr1", oskar_mem_char(names[0]));
        ASSERT_STREQ("dataset_attr2", oskar_mem_char(names[1]));
        ASSERT_DOUBLE_EQ(2.99792458e8, oskar_mem_double(values[0], &status)[0]);
        ASSERT_DOUBLE_EQ(6.63e-34, oskar_mem_double(values[1], &status)[0]);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Print the attributes.
        print_attributes(num_attr, names, values, &status);

        // Read all the attributes on the root group.
        oskar_hdf5_read_attributes(
                h, "/", &num_attr, &names, &values, &status
        );
        print_attributes(num_attr, names, values, &status);

        // Read all the attributes on Group1.
        oskar_hdf5_read_attributes(
                h, "/Group1", &num_attr, &names, &values, &status
        );
        print_attributes(num_attr, names, values, &status);

        // Read all the attributes on Subgroup1.
        oskar_hdf5_read_attributes(
                h, "/Group1/Subgroup1", &num_attr, &names, &values, &status
        );
        print_attributes(num_attr, names, values, &status);

        // Try with invalid argument.
        {
            int status = 0;
            oskar_hdf5_read_attributes(
                    h, "/Group1", 0, &names, &values, &status
            );
            ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
        }

        // Clean up.
        for (int i = 0; i < num_attr; ++i)
        {
            oskar_mem_free(names[i], &status);
            oskar_mem_free(values[i], &status);
        }
        free(names);
        free(values);
        oskar_hdf5_close(h);
    }

    // Overwrite the attributes.
    {
        int status = 0;
        oskar_HDF5* h = oskar_hdf5_open(filename, 'a', &status);

        // Write new values after checking old ones.
        {
            const char* name = "int_attr1";
            int val = oskar_hdf5_read_attribute_int(h, 0, name, &status);
            ASSERT_EQ(42, val);
            int new_val = 100;
            oskar_hdf5_write_attribute_int(h, 0, name, new_val, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            val = oskar_hdf5_read_attribute_int(h, 0, name, &status);
            ASSERT_EQ(new_val, val);
        }
        {
            const char* name = "dbl_attr1";
            double val = oskar_hdf5_read_attribute_double(h, 0, name, &status);
            ASSERT_DOUBLE_EQ(3.14, val);
            double new_val = 987.654;
            oskar_hdf5_write_attribute_double(h, 0, name, new_val, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            val = oskar_hdf5_read_attribute_double(h, 0, name, &status);
            ASSERT_DOUBLE_EQ(new_val, val);
        }
        {
            const char* name = "str_attr1";
            const char* new_val = "The quick brown fox jumps over the lazy dog";
            oskar_hdf5_write_attribute_string(h, 0, name, new_val, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            char* val = oskar_hdf5_read_attribute_string(h, 0, name, &status);
            ASSERT_STREQ(new_val, val);
            free(val);
        }
        oskar_hdf5_close(h);
    }
    remove(filename);
}


TEST(HDF5, test_error_non_existent_file)
{
    // Test trying to open a non-existent file for reading.
    int status = 0;
    oskar_HDF5* h = oskar_hdf5_open(
            "a_file_that_does_not_exist.h5", 'r', &status
    );
    ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
    oskar_hdf5_close(h);
}


TEST(HDF5, test_error_non_existent_group)
{
    // Test trying to use a non-existent parent group.
    int status = 0;
    const char* filename = "temp_test_hdf5_no_parent_group.h5";
    oskar_HDF5* h = oskar_hdf5_open(filename, 'w', &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write some test data.
    oskar_Mem* data = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 10, &status);
    oskar_hdf5_write_dataset(h, 0, "test", 0, 0, data, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try to write attributes to a non-existent group.
    {
        int status = 0;
        printf("Error expected: writing double to non-existent group\n");
        oskar_hdf5_write_attribute_double(
                h, "group_not_found", "dbl_attr", 1.2, &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
    }
    {
        int status = 0;
        printf("Error expected: writing int to non-existent group\n");
        oskar_hdf5_write_attribute_int(
                h, "group_not_found", "int_attr", -1, &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
    }
    {
        int status = 0;
        printf("Error expected: writing string to non-existent group\n");
        oskar_hdf5_write_attribute_string(
                h, "group_not_found", "str_attr", "won't work", &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
    }

    // Try to read attributes from a non-existent group.
    {
        int status = 0;
        printf("Error expected: reading double from non-existent group\n");
        double val = oskar_hdf5_read_attribute_double(
                h, "group_not_found", "dbl_attr", &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
        ASSERT_DOUBLE_EQ(0.0, val);
    }
    {
        int status = 0;
        printf("Error expected: reading int from non-existent group\n");
        int val = oskar_hdf5_read_attribute_int(
                h, "group_not_found", "int_attr", &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
        ASSERT_EQ(0, val);
    }
    {
        int status = 0;
        printf("Error expected: reading string from non-existent group\n");
        const char* val = oskar_hdf5_read_attribute_string(
                h, "group_not_found", "str_attr", &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
        ASSERT_EQ(NULL, val);
    }

    // Try to write a dataset to a non-existent group.
    {
        int status = 0;
        printf("Error expected: writing dataset to non-existent group\n");
        oskar_hdf5_write_dataset(
                h, "not_a_group", "test2", 0, 0, data, 0, &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
    }

    // Try to read a dataset from a non-existent group.
    {
        int status = 0;
        printf("Error expected: reading dataset from non-existent group\n");
        oskar_Mem* data = oskar_hdf5_read_dataset(
                h, "not_a_group", "test", 0, 0, &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
        ASSERT_EQ(NULL, data);
    }
    {
        int status = 0;
        int num_dims = 0;
        size_t* dims = 0;
        printf("Error expected: reading dimensions from non-existent group\n");
        oskar_hdf5_read_dataset_dims(
                h, "not_a_group", "test", &num_dims, &dims, &status
        );
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
        ASSERT_EQ(0, num_dims);
        ASSERT_EQ(NULL, dims);
    }

    // Try to create a group with a non-existent parent.
    {
        int status = 0;
        printf("Error expected: creating group with non-existent parent\n");
        oskar_hdf5_write_group(h, "not_a_group_either", "new_group", &status);
        ASSERT_EQ(OSKAR_ERR_FILE_IO, status);
    }

    // Clean up.
    oskar_hdf5_close(h);
    oskar_mem_free(data, &status);
    remove(filename);
}


TEST(HDF5, test_error_wrong_attribute_types)
{
    // Test trying to read attributes with the wrong data type.
    int status = 0;
    const char* filename = "temp_test_hdf5_wrong_attribute_type.h5";
    oskar_HDF5* h = oskar_hdf5_open(filename, 'w', &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write some attributes.
    oskar_hdf5_write_attribute_double(h, "/", "a_double", 123.456, &status);
    oskar_hdf5_write_attribute_int(h, "/", "an_int", 789, &status);
    oskar_hdf5_write_attribute_string(h, "/", "a_string", "hello", &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try to read attributes with the wrong type.
    {
        int status = 0;
        printf("Error expected: reading an integer as a double\n");
        double val = oskar_hdf5_read_attribute_double(
                h, 0, "an_int", &status
        );
        ASSERT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        ASSERT_DOUBLE_EQ(0.0, val);
    }
    {
        int status = 0;
        printf("Error expected: reading a double as an integer\n");
        int val = oskar_hdf5_read_attribute_int(
                h, "/", "a_double", &status
        );
        ASSERT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        ASSERT_EQ(0, val);
    }
    {
        int status = 0;
        printf("Error expected: reading a double as a string\n");
        char* val = oskar_hdf5_read_attribute_string(
                h, "/", "a_double", &status
        );
        ASSERT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        ASSERT_EQ(NULL, val);
    }

    // Clean up.
    oskar_hdf5_close(h);
    remove(filename);
}
