/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_hdf5.h"
#include "utility/private_hdf5.h"

#ifdef OSKAR_HAVE_HDF5
#include "hdf5.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OSKAR_HAVE_HDF5
static int iter_func(hid_t loc_id, const char* name, const H5O_info_t* info,
            void* operator_data);
#endif

oskar_HDF5* oskar_hdf5_open(const char* file_path, int* status)
{
    oskar_HDF5* h = (oskar_HDF5*) calloc(1, sizeof(oskar_HDF5));
#ifdef OSKAR_HAVE_HDF5
    /* Open the HDF5 file for reading. */
    h->file_id = H5Fopen(file_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h->file_id < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_error(0, "Error opening HDF5 file '%s'\n", file_path);
        free(h);
        return 0;
    }

    /* Iterate all datasets in the file using the callback function. */
    H5Ovisit3(h->file_id, H5_INDEX_NAME, H5_ITER_NATIVE, iter_func, h,
            H5O_INFO_BASIC);
#else
    (void)file_path;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
    return h;
}

#ifdef OSKAR_HAVE_HDF5
int iter_func(hid_t loc_id, const char* name, const H5O_info_t* info,
            void* operator_data)
{
    (void)loc_id;
    if (name[0] == '.') return 0; /* Skip root group. */

    oskar_HDF5* h = (oskar_HDF5*) operator_data;
    const int name_len = (int) strlen(name);
    if (name_len == 0) return 0;
    switch (info->type)
    {
        case H5O_TYPE_GROUP:
            break;
        case H5O_TYPE_DATASET:
        {
            const int i = h->num_datasets++;
            h->names = (char**) realloc(h->names, (i + 1) * sizeof(char*));
            h->names[i] = (char*) calloc(2 + name_len, sizeof(char));
            h->names[i][0] = '/';
            memcpy(&h->names[i][1], name, name_len);
            break;
        }
        case H5O_TYPE_NAMED_DATATYPE:
            break;
        default:
            break;
    }
    return 0;
}
#endif


void oskar_hdf5_close(oskar_HDF5* h)
{
    if (!h) return;
#ifdef OSKAR_HAVE_HDF5
    (void) H5Fclose(h->file_id);
#endif
    for (int i = 0; i < h->num_datasets; ++i)
        free(h->names[i]);
    free(h->names);
    free(h);
}


const char* oskar_hdf5_dataset_name(const oskar_HDF5* h, int i)
{
    return h->names[i];
}


int oskar_hdf5_num_datasets(const oskar_HDF5* h)
{
    return h->num_datasets;
}


#ifdef OSKAR_HAVE_HDF5
static oskar_Mem* read_hyperslab(oskar_HDF5* h, const hid_t dataset,
        int num_dims, const size_t* offset, const size_t* size, int* status)
{
    hid_t memspace = H5S_ALL;
    hid_t filespace = H5S_ALL;
    herr_t hdf5_error = 0;
    size_t num_elements = 0;
    oskar_Mem* data = 0;
    if (*status || !h) return 0;

    /* Get dataset metadata. */
    const hid_t dataspace = H5Dget_space(dataset);
    const hid_t datatype = H5Dget_type(dataset);

    /* Select dimensions if given. */
    if (num_dims > 0)
    {
        /* Define hyperslab in the dataset to read. */
        hsize_t count_out = 1, *count_in = 0, *offset_in = 0;
        count_in = (hsize_t*) calloc(num_dims, sizeof(size_t));
        offset_in = (hsize_t*) calloc(num_dims, sizeof(size_t));
        filespace = dataspace;
        for (int i = 0; i < num_dims; ++i)
        {
            offset_in[i] = offset[i];
            count_in[i] = (size[i] > 0) ? size[i] : 1;
        }
        if (hdf5_error >= 0)
            hdf5_error = (int) H5Sselect_hyperslab(filespace,
                    H5S_SELECT_SET, offset_in, NULL, count_in, NULL);

        /* Define a 1D memory dataspace big enough to hold everything. */
        const hsize_t offset_out = 0;
        for (int i = 0; i < num_dims; ++i)
            count_out *= ((size[i] > 0) ? size[i] : 1);
        memspace = H5Screate_simple(1, &count_out, NULL);
        if (hdf5_error >= 0)
            hdf5_error = (int) H5Sselect_hyperslab(memspace,
                    H5S_SELECT_SET, &offset_out, NULL, &count_out, NULL);
        free(count_in);
        free(offset_in);

        /* Set the size of the array to return. */
        num_elements = count_out;
    }
    else
    {
        /* Set the size of the array to return. */
        num_elements = (size_t)H5Sget_simple_extent_npoints(dataspace);
    }

    /* Read data from hyperslab in the file into the hyperslab in memory. */
    if (hdf5_error >= 0)
    {
        switch (H5Tget_class(datatype))
        {
        case H5T_INTEGER:
        {
            data = oskar_mem_create(OSKAR_INT, OSKAR_CPU, num_elements, status);
            if (!*status)
                hdf5_error = H5Dread(dataset, H5T_NATIVE_INT, memspace,
                        filespace, H5P_DEFAULT, oskar_mem_void(data));
            break;
        }
        case H5T_FLOAT:
        {
            const size_t size0 = H5Tget_size(datatype);
            if (size0 <= sizeof(float))
            {
                data = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU,
                        num_elements, status);
                if (!*status)
                    hdf5_error = H5Dread(dataset, H5T_NATIVE_FLOAT, memspace,
                            filespace, H5P_DEFAULT, oskar_mem_void(data));
            }
            else if (size0 >= sizeof(double))
            {
                data = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
                        num_elements, status);
                if (!*status)
                    hdf5_error = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace,
                            filespace, H5P_DEFAULT, oskar_mem_void(data));
            }
            break;
        }
        case H5T_COMPOUND:
        {
            if (H5Tget_nmembers(datatype) == 2)
            {
                const H5T_class_t class0 = H5Tget_member_class(datatype, 0);
                const H5T_class_t class1 = H5Tget_member_class(datatype, 1);
                const hid_t type0 = H5Tget_member_type(datatype, 0);
                const hid_t type1 = H5Tget_member_type(datatype, 1);
                const size_t size0 = H5Tget_size(type0);
                const size_t size1 = H5Tget_size(type1);
                if (class0 == H5T_FLOAT && class0 == class1 && size0 == size1)
                {
                    if (size0 == sizeof(float))
                    {
                        data = oskar_mem_create(OSKAR_SINGLE_COMPLEX,
                                OSKAR_CPU, num_elements, status);
                        if (!*status)
                            hdf5_error = H5Dread(dataset, datatype, memspace,
                                    filespace, H5P_DEFAULT,
                                    oskar_mem_void(data));
                    }
                    else if (size0 == sizeof(double))
                    {
                        data = oskar_mem_create(OSKAR_DOUBLE_COMPLEX,
                                OSKAR_CPU, num_elements, status);
                        if (!*status)
                            hdf5_error = H5Dread(dataset, datatype, memspace,
                                    filespace, H5P_DEFAULT,
                                    oskar_mem_void(data));
                    }
                    else
                    {
                        *status = OSKAR_ERR_BAD_DATA_TYPE;
                        oskar_log_error(0, "Unknown HDF5 complex format.");
                    }
                }
                else
                {
                    *status = OSKAR_ERR_BAD_DATA_TYPE;
                    oskar_log_error(0,
                            "Need matching float types in HDF5 struct.");
                }
                H5Tclose(type0);
                H5Tclose(type1);
            }
            else
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                oskar_log_error(0, "Need exactly 2 elements in HDF5 struct.");
            }
            break;
        }
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            oskar_log_error(0, "Unknown HDF5 datatype for dataset.");
        }
    }

    /* Close/release resources. */
    H5Tclose(datatype);
    H5Sclose(dataspace);
    if (memspace) H5Sclose(memspace);

    /* Return the data. */
    if (!*status && hdf5_error < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_error(0, "HDF5 error, code %d", hdf5_error);
    }
    return data;
}

static void read_dims(oskar_HDF5* h, const hid_t dataset,
        int* num_dims, size_t** dims, int* status)
{
    if (*status || !h) return;
    const hid_t dataspace = H5Dget_space(dataset);
    const int num_dims_l = H5Sget_simple_extent_ndims(dataspace);
    if (dims)
    {
        hsize_t* dims_l = (hsize_t*) calloc(num_dims_l, sizeof(hsize_t));
        H5Sget_simple_extent_dims(dataspace, dims_l, NULL);
        *dims = (size_t*) realloc(*dims, num_dims_l * sizeof(size_t));
        for (int i = 0; i < num_dims_l; ++i)
            (*dims)[i] = (size_t) (dims_l[i]);
        free(dims_l);
    }
    if (num_dims) *num_dims = num_dims_l;
    H5Sclose(dataspace);
}
#endif


void oskar_hdf5_read_dataset_dims(oskar_HDF5* h, const char* dataset_path,
        int* num_dims, size_t** dims, int* status)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !h) return;

    /* Open the dataset. */
    const hid_t dataset = H5Dopen2(h->file_id, dataset_path, H5P_DEFAULT);
    if (dataset < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_error(0, "Error opening dataset '%s'\n", dataset_path);
        return;
    }

    /* Read dataset dimensions. */
    read_dims(h, dataset, num_dims, dims, status);

    /* Close/release resources. */
    H5Dclose(dataset);
#else
    (void)dataset_path;
    (void)num_dims;
    (void)dims;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
    return;
#endif
}


oskar_Mem* oskar_hdf5_read_dataset(oskar_HDF5* h, const char* dataset_path,
        int* num_dims, size_t** dims, int* status)
{
#ifdef OSKAR_HAVE_HDF5
    oskar_Mem* data = 0;
    if (*status || !h) return 0;

    /* Open the dataset. */
    const hid_t dataset = H5Dopen2(h->file_id, dataset_path, H5P_DEFAULT);
    if (dataset < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_error(0, "Error opening dataset '%s'\n", dataset_path);
        return 0;
    }

    /* Read the whole dataset. */
    data = read_hyperslab(h, dataset, -1, 0, 0, status);

    /* Read dataset dimensions. */
    read_dims(h, dataset, num_dims, dims, status);

    /* Close/release resources. */
    H5Dclose(dataset);

    /* Return the data. */
    return data;
#else
    (void)dataset_path;
    (void)num_dims;
    (void)dims;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
    return 0;
#endif
}


oskar_Mem* oskar_hdf5_read_hyperslab(oskar_HDF5* h, const char* dataset_path,
        int num_dims, const size_t* offset, const size_t* size, int* status)
{
#ifdef OSKAR_HAVE_HDF5
    oskar_Mem* data = 0;
    if (*status || !h) return 0;

    /* Open the dataset. */
    const hid_t dataset = H5Dopen2(h->file_id, dataset_path, H5P_DEFAULT);
    if (dataset < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_error(0, "Error opening dataset '%s'\n", dataset_path);
        return 0;
    }

    /* Read hyperslab. */
    data = read_hyperslab(h, dataset, num_dims, offset, size, status);

    /* Close/release resources. */
    H5Dclose(dataset);

    /* Return the data. */
    return data;
#else
    (void)dataset_path;
    (void)num_dims;
    (void)offset;
    (void)size;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
    return 0;
#endif
}


#ifdef __cplusplus
}
#endif
