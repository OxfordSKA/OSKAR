/*
 * Copyright (c) 2020-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_hdf5.h"
#include "utility/private_hdf5.h"

#ifdef OSKAR_HAVE_HDF5
#include "hdf5.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OSKAR_HAVE_HDF5
static int iter_func(
        hid_t loc_id,
        const char* name,
        const H5O_info_t* info,
        void* operator_data
);
#endif


oskar_HDF5* oskar_hdf5_open(const char* file_path, char mode, int* status)
{
    oskar_HDF5* handle = (oskar_HDF5*) calloc(1, sizeof(oskar_HDF5));
    handle->mutex = oskar_mutex_create();
    handle->refcount++;
#ifdef OSKAR_HAVE_HDF5
    if (mode == 'w')
    {
        /* Create a new HDF5 file. */
        handle->file_id = H5Fcreate(
                file_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT
        );
        if (handle->file_id < 0)
        {
            *status = OSKAR_ERR_FILE_IO;                  /* LCOV_EXCL_LINE */
            oskar_hdf5_close(handle);                     /* LCOV_EXCL_LINE */
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Error creating HDF5 file '%s'.", file_path
            );
            return 0;                                     /* LCOV_EXCL_LINE */
        }
    }
    else if (mode == 'r')
    {
        /* Check to make sure the file exists. */
        if (!oskar_file_exists(file_path))
        {
            *status = OSKAR_ERR_FILE_IO;
            oskar_hdf5_close(handle);
            oskar_log_error(
                    0, "HDF5 file '%s' does not exist.", file_path
            );
            return 0;
        }

        /* Open the HDF5 file for reading. */
        handle->file_id = H5Fopen(file_path, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (handle->file_id < 0)
        {
            *status = OSKAR_ERR_FILE_IO;                  /* LCOV_EXCL_LINE */
            oskar_hdf5_close(handle);                     /* LCOV_EXCL_LINE */
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Error opening HDF5 file '%s'.", file_path
            );
            return 0;                                     /* LCOV_EXCL_LINE */
        }

        /* Iterate all datasets in the file using the callback function. */
#if H5_VERSION_GE(1, 12, 0)
        H5Ovisit(
                handle->file_id, H5_INDEX_NAME, H5_ITER_NATIVE,
                iter_func, handle, H5O_INFO_BASIC
        );
#else
        H5Ovisit(
                handle->file_id, H5_INDEX_NAME, H5_ITER_NATIVE,
                iter_func, handle
        );
#endif
    }
    else if (mode == 'a')
    {
        /* Open the HDF5 file for read/write. */
        if (oskar_file_exists(file_path))
        {
            handle->file_id = H5Fopen(file_path, H5F_ACC_RDWR, H5P_DEFAULT);
            if (handle->file_id < 0)
            {
                *status = OSKAR_ERR_FILE_IO;              /* LCOV_EXCL_LINE */
                oskar_hdf5_close(handle);                 /* LCOV_EXCL_LINE */
                oskar_log_error(                          /* LCOV_EXCL_LINE */
                        0, "Error opening HDF5 file '%s'.", file_path
                );
                return 0;                                 /* LCOV_EXCL_LINE */
            }

            /* Iterate all datasets in the file using the callback function. */
#if H5_VERSION_GE(1, 12, 0)
            H5Ovisit(
                    handle->file_id, H5_INDEX_NAME, H5_ITER_NATIVE,
                    iter_func, handle, H5O_INFO_BASIC
            );
#else
            H5Ovisit(
                    handle->file_id, H5_INDEX_NAME, H5_ITER_NATIVE,
                    iter_func, handle
            );
#endif
        }
        else
        {
            handle->file_id = H5Fcreate(
                    file_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT
            );
            if (handle->file_id < 0)
            {
                *status = OSKAR_ERR_FILE_IO;              /* LCOV_EXCL_LINE */
                oskar_hdf5_close(handle);                 /* LCOV_EXCL_LINE */
                oskar_log_error(                          /* LCOV_EXCL_LINE */
                        0, "Error creating HDF5 file '%s'.", file_path
                );
                return 0;                                 /* LCOV_EXCL_LINE */
            }
        }
    }
#else
    (void) file_path;
    (void) mode;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
    return handle;
}


#ifdef OSKAR_HAVE_HDF5
int iter_func(
        hid_t loc_id,
        const char* name,
        const H5O_info_t* info,
        void* operator_data
)
{
    (void) loc_id;
    if (name[0] == '.') return 0; /* Skip root group. */

    oskar_HDF5* handle = (oskar_HDF5*) operator_data;
    const size_t name_len = strlen(name);
    if (name_len == 0) return 0;
    switch (info->type)
    {
        case H5O_TYPE_GROUP:
            break;
        case H5O_TYPE_DATASET:
        {
            const int i = handle->num_datasets++;
            handle->names = (char**) realloc(
                    handle->names, (i + 1) * sizeof(char*)
            );
            handle->names[i] = (char*) calloc(2 + name_len, sizeof(char));
            handle->names[i][0] = '/';
            memcpy(&handle->names[i][1], name, name_len);
            break;
        }
        default:                                          /* LCOV_EXCL_LINE */
            break;                                        /* LCOV_EXCL_LINE */
    }
    return 0;
}
#endif


void oskar_hdf5_close(oskar_HDF5* handle)
{
    if (!handle) return;
    oskar_mutex_lock(handle->mutex);
    handle->refcount--;
    oskar_mutex_unlock(handle->mutex);
    if (handle->refcount <= 0)
    {
        oskar_mutex_free(handle->mutex);
#ifdef OSKAR_HAVE_HDF5
        if (handle->file_id > 0)
        {
            (void) H5Fclose(handle->file_id);
        }
#endif
        for (int i = 0; i < handle->num_datasets; ++i)
        {
            free(handle->names[i]);
        }
        free(handle->names);
        free(handle);
    }
}


void oskar_hdf5_ref_inc(oskar_HDF5* handle)
{
    if (!handle) return;
    oskar_mutex_lock(handle->mutex);
    handle->refcount++;
    oskar_mutex_unlock(handle->mutex);
}


int oskar_hdf5_dataset_exists(const oskar_HDF5* handle, const char* name)
{
    for (int i = 0; i < handle->num_datasets; ++i)
    {
        if (!strcmp(name, handle->names[i]))
        {
            return 1;
        }
    }
    return 0;
}


#ifdef OSKAR_HAVE_HDF5
static void get_data_dims(
        int num_dims,
        const size_t* offset,
        const size_t* dims,
        const oskar_Mem* data,
        int* actual_num_dims,
        hsize_t** actual_offset,
        hsize_t** actual_dims
)
{
    /*
     * Get the actual dimensions of the data.
     * Allow for invalid inputs, and add an extra dimension
     * of length 4 for "matrix" types.
     */
    size_t alloc_dims = ((num_dims > 0) ? num_dims + 1 : 2);
    *actual_offset = (hsize_t*) calloc(alloc_dims, sizeof(hsize_t));
    *actual_dims = (hsize_t*) calloc(alloc_dims, sizeof(hsize_t));
    *actual_num_dims = (num_dims > 0 ? num_dims : 1);
    if (offset && num_dims > 0)
    {
        for (int i = 0; i < num_dims; ++i)
        {
            (*actual_offset)[i] = offset[i];
        }
    }
    else
    {
        (*actual_offset)[0] = 0;
    }
    if (dims && num_dims > 0)
    {
        for (int i = 0; i < num_dims; ++i)
        {
            (*actual_dims)[i] = dims[i];
        }
    }
    else
    {
        (*actual_dims)[0] = oskar_mem_length(data);
    }
    if (oskar_mem_is_matrix(data))
    {
        (*actual_offset)[*actual_num_dims] = 0;
        (*actual_dims)[*actual_num_dims] = 4;
        (*actual_num_dims)++;
    }
}


static hid_t get_data_type(const oskar_Mem* data, int* status)
{
    hid_t type_id = 0;
    switch (oskar_mem_type(data))
    {
    case OSKAR_CHAR:
        type_id = H5Tcopy(H5T_NATIVE_CHAR);
        break;
    case OSKAR_INT:
        type_id = H5Tcopy(H5T_NATIVE_INT);
        break;
    case OSKAR_SINGLE:
        type_id = H5Tcopy(H5T_NATIVE_FLOAT);
        break;
    case OSKAR_DOUBLE:
        type_id = H5Tcopy(H5T_NATIVE_DOUBLE);
        break;
    case OSKAR_SINGLE_COMPLEX:
    case OSKAR_SINGLE_COMPLEX_MATRIX:
        type_id = H5Tcreate(H5T_COMPOUND, 2 * sizeof(float));
        H5Tinsert(type_id, "r", 0, H5T_NATIVE_FLOAT);
        H5Tinsert(type_id, "i", sizeof(float), H5T_NATIVE_FLOAT);
        break;
    case OSKAR_DOUBLE_COMPLEX:
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
        type_id = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
        H5Tinsert(type_id, "r", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(type_id, "i", sizeof(double), H5T_NATIVE_DOUBLE);
        break;
    default:                                              /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_BAD_DATA_TYPE;                /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Unknown array data type for HDF5 file."
        );
    }
    return type_id;
}


static hid_t get_object(
        const oskar_HDF5* handle,
        const char* object_path,
        int* object_is_root,
        int* status
)
{
    hid_t obj_id = -1;
    *object_is_root = (
            !object_path ||
            strcmp(object_path, "/") == 0 ||
            strcmp(object_path, "") == 0
    );
    if (*object_is_root)
    {
        obj_id = handle->file_id;
    }
    else if (H5Lexists(handle->file_id, object_path, H5P_DEFAULT))
    {
        obj_id = H5Oopen(handle->file_id, object_path, H5P_DEFAULT);
    }
    if (obj_id < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_log_error(
                0, "Error opening HDF5 object '%s'.", object_path
        );
    }
    return obj_id;
}


static hid_t open_or_create_dataset(
        oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int num_dims,
        const hsize_t* dims,
        const oskar_Mem* data,
        int* status
)
{
    if (*status || !handle) return 0;

    /* Get parent object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, parent_path, &object_is_root, status
    );
    if (*status || obj_id < 0) return -1;

    /* Get the HDF5 data type. */
    const hid_t type_id = get_data_type(data, status);

    /* Open or create the dataset as needed. */
    hid_t dataset_id = 0;
    if (H5Lexists(obj_id, dataset_name, H5P_DEFAULT))
    {
        dataset_id = H5Dopen(obj_id, dataset_name, H5P_DEFAULT);
        if (dataset_id < 0)
        {
            *status = OSKAR_ERR_FILE_IO;                  /* LCOV_EXCL_LINE */
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Error opening HDF5 dataset '%s' in '%s'.",
                    dataset_name, parent_path
            );
        }
    }
    else
    {
        const hid_t filespace_id = H5Screate_simple(num_dims, dims, NULL);
        dataset_id = H5Dcreate(
                obj_id, dataset_name, type_id, filespace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
        );
        if (dataset_id < 0)
        {
            *status = OSKAR_ERR_FILE_IO;                  /* LCOV_EXCL_LINE */
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Error creating HDF5 dataset '%s' in '%s'.",
                    dataset_name, parent_path
            );
        }
        H5Sclose(filespace_id);
    }

    /* Close/release resources. */
    H5Tclose(type_id);
    if (!object_is_root) H5Oclose(obj_id);
    return dataset_id;
}


static oskar_Mem* read_hyperslab(
        const oskar_HDF5* handle,
        const hid_t dataset,
        int num_dims,
        const size_t* offset,
        const size_t* size,
        int* status
)
{
    hid_t memspace = H5S_ALL;
    hid_t filespace = H5S_ALL;
    herr_t hdf5_error = 0;
    size_t num_elements = 0;
    oskar_Mem* data = 0;
    if (*status || !handle) return 0;

    /* Get dataset metadata. */
    const hid_t dataspace = H5Dget_space(dataset);
    const hid_t datatype = H5Dget_type(dataset);

    /* Select dimensions if given. */
    if (num_dims > 0)
    {
        /* Define hyperslab in the dataset to read. */
        hsize_t count_out = 1, *count_in = 0, *offset_in = 0;
        count_in = (hsize_t*) calloc(num_dims, sizeof(hsize_t));
        offset_in = (hsize_t*) calloc(num_dims, sizeof(hsize_t));
        filespace = dataspace;
        for (int i = 0; i < num_dims; ++i)
        {
            offset_in[i] = offset[i];
            count_in[i] = (size[i] > 0) ? size[i] : 1;
        }
        if (hdf5_error >= 0)
        {
            hdf5_error = H5Sselect_hyperslab(
                    filespace, H5S_SELECT_SET, offset_in, NULL, count_in, NULL
            );
        }

        /* Define a 1D memory dataspace big enough to hold everything. */
        const hsize_t offset_out = 0;
        for (int i = 0; i < num_dims; ++i)
        {
            count_out *= ((size[i] > 0) ? size[i] : 1);
        }
        memspace = H5Screate_simple(1, &count_out, NULL);
        if (hdf5_error >= 0)
        {
            hdf5_error = H5Sselect_hyperslab(
                    memspace, H5S_SELECT_SET,
                    &offset_out, NULL, &count_out, NULL
            );
        }
        free(count_in);
        free(offset_in);

        /* Set the size of the array to return. */
        num_elements = count_out;
    }
    else
    {
        /* Set the size of the array to return. */
        num_elements = (size_t) H5Sget_simple_extent_npoints(dataspace);
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
            {
                hdf5_error = H5Dread(
                        dataset, H5T_NATIVE_INT, memspace,
                        filespace, H5P_DEFAULT, oskar_mem_void(data)
                );
            }
            break;
        }
        case H5T_FLOAT:
        {
            const size_t size0 = H5Tget_size(datatype);
            if (size0 <= sizeof(float))
            {
                data = oskar_mem_create(
                        OSKAR_SINGLE, OSKAR_CPU, num_elements, status
                );
                if (!*status)
                {
                    hdf5_error = H5Dread(
                            dataset, H5T_NATIVE_FLOAT, memspace, filespace,
                            H5P_DEFAULT, oskar_mem_void(data)
                    );
                }
            }
            else if (size0 >= sizeof(double))
            {
                data = oskar_mem_create(
                        OSKAR_DOUBLE, OSKAR_CPU, num_elements, status
                );
                if (!*status)
                {
                    hdf5_error = H5Dread(
                            dataset, H5T_NATIVE_DOUBLE, memspace, filespace,
                            H5P_DEFAULT, oskar_mem_void(data)
                    );
                }
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
                        data = oskar_mem_create(
                                OSKAR_SINGLE_COMPLEX, OSKAR_CPU,
                                num_elements, status
                        );
                        if (!*status)
                        {
                            hdf5_error = H5Dread(
                                    dataset, datatype, memspace, filespace,
                                    H5P_DEFAULT, oskar_mem_void(data)
                            );
                        }
                    }
                    else if (size0 == sizeof(double))
                    {
                        data = oskar_mem_create(OSKAR_DOUBLE_COMPLEX,
                                OSKAR_CPU, num_elements, status);
                        if (!*status)
                        {
                            hdf5_error = H5Dread(
                                    dataset, datatype, memspace, filespace,
                                    H5P_DEFAULT, oskar_mem_void(data)
                            );
                        }
                    }
                    else
                    {
                        *status = OSKAR_ERR_BAD_DATA_TYPE; /* LCOV_EXCL_LINE */
                        oskar_log_error(                   /* LCOV_EXCL_LINE */
                                0, "Unknown HDF5 complex format."
                        );
                    }
                }
                else
                {
                    *status = OSKAR_ERR_BAD_DATA_TYPE;    /* LCOV_EXCL_LINE */
                    oskar_log_error(                      /* LCOV_EXCL_LINE */
                            0, "Need matching float types in HDF5 struct."
                    );
                }
                H5Tclose(type0);
                H5Tclose(type1);
            }
            else
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;        /* LCOV_EXCL_LINE */
                oskar_log_error(                          /* LCOV_EXCL_LINE */
                        0, "Need exactly 2 elements in HDF5 struct."
                );
            }
            break;
        }
        default:                                          /* LCOV_EXCL_LINE */
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Unknown HDF5 datatype for requested dataset."
            );
        }
    }

    /* Close/release resources. */
    H5Tclose(datatype);
    H5Sclose(dataspace);
    if (memspace) H5Sclose(memspace);

    /* Check for errors and return the data. */
    if (!*status && hdf5_error < 0)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error reading HDF5 hyperslab (code %d).", hdf5_error
        );
    }
    return data;
}


static void read_dims(
        const oskar_HDF5* handle,
        const hid_t dataset_id,
        int* num_dims,
        size_t** dims,
        const int* status
)
{
    if (*status || !handle) return;
    const hid_t dataspace_id = H5Dget_space(dataset_id);
    const int num_dims_l = H5Sget_simple_extent_ndims(dataspace_id);
    if (dims)
    {
        hsize_t* dims_l = (hsize_t*) calloc(num_dims_l, sizeof(hsize_t));
        H5Sget_simple_extent_dims(dataspace_id, dims_l, NULL);
        *dims = (size_t*) realloc(*dims, num_dims_l * sizeof(size_t));
        for (int i = 0; i < num_dims_l; ++i)
        {
            (*dims)[i] = (size_t) (dims_l[i]);
        }
        free(dims_l);
    }
    if (num_dims) *num_dims = num_dims_l;
    H5Sclose(dataspace_id);
}


static void write_hyperslab(
        const oskar_HDF5* handle,
        hid_t dataset_id,
        int num_dims,
        const hsize_t* offset,
        const hsize_t* size,
        const oskar_Mem* data,
        int* status
)
{
    if (*status || !handle || dataset_id < 0) return;

    /* Get the HDF5 data type. */
    const hid_t type_id = get_data_type(data, status);

    /* Get dataspace and select hyperslab. */
    const hid_t filespace_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(
            filespace_id, H5S_SELECT_SET, offset, NULL, size, NULL
    );

    /* Create memory dataspace. */
    const hid_t memspace_id = H5Screate_simple(num_dims, size, NULL);

    /* Write the data. */
    const herr_t hdf5_error = H5Dwrite(
            dataset_id, type_id, memspace_id, filespace_id, H5P_DEFAULT,
            oskar_mem_void_const(data)
    );
    if (hdf5_error < 0)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error writing HDF5 hyperslab (code %d).", hdf5_error
        );
    }

    /* Close/release resources. */
    H5Sclose(memspace_id);
    H5Sclose(filespace_id);
    H5Tclose(type_id);
}
#endif


int oskar_hdf5_read_attribute_int(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int* status
)
{
    int value = 0;
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return value;
    oskar_mutex_lock(handle->mutex);

    /* Get the object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, object_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 object for integer attribute '%s' on '%s'.",
                attribute_name, object_path
        );
        return value;
    }

    /* Get the attribute ID and read it. */
    const hid_t attr_id = H5Aopen(obj_id, attribute_name, H5P_DEFAULT);
    const hid_t dataspace_id = H5Aget_space(attr_id);
    const hid_t datatype_id = H5Aget_type(attr_id);
    const size_t len = (size_t) H5Sget_simple_extent_npoints(dataspace_id);
    if (len == 1 && H5Tget_class(datatype_id) == H5T_INTEGER)
    {
        H5Aread(attr_id, H5T_NATIVE_INT, &value);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        oskar_log_error(
                0, "HDF5 attribute '%s' on '%s' is not a single integer",
                attribute_name, object_path
        );
    }

    /* Close/release resources. */
    H5Aclose(attr_id);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) object_path;
    (void) attribute_name;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
    return value;
}


double oskar_hdf5_read_attribute_double(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int* status
)
{
    double value = 0.0;
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return value;
    oskar_mutex_lock(handle->mutex);

    /* Get the object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, object_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 object for double attribute '%s' on '%s'.",
                attribute_name, object_path
        );
        return value;
    }

    /* Get the attribute ID and read it. */
    const hid_t attr_id = H5Aopen(obj_id, attribute_name, H5P_DEFAULT);
    const hid_t dataspace_id = H5Aget_space(attr_id);
    const hid_t datatype_id = H5Aget_type(attr_id);
    const size_t len = (size_t) H5Sget_simple_extent_npoints(dataspace_id);
    if (len == 1 && H5Tget_class(datatype_id) == H5T_FLOAT)
    {
        H5Aread(attr_id, H5T_NATIVE_DOUBLE, &value);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        oskar_log_error(
                0, "HDF5 attribute '%s' on '%s' is not a single float",
                attribute_name, object_path
        );
    }

    /* Close/release resources. */
    H5Aclose(attr_id);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) object_path;
    (void) attribute_name;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
    return value;
}


char* oskar_hdf5_read_attribute_string(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int* status
)
{
    char* value = 0;
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return value;
    oskar_mutex_lock(handle->mutex);

    /* Get the object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, object_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 object for string attribute '%s' on '%s'.",
                attribute_name, object_path
        );
        return value;
    }

    /* Get the attribute ID and read it. */
    const hid_t attr_id = H5Aopen(obj_id, attribute_name, H5P_DEFAULT);
    const hid_t dataspace_id = H5Aget_space(attr_id);
    const hid_t datatype_id = H5Aget_type(attr_id);
    const size_t len = (size_t) H5Sget_simple_extent_npoints(dataspace_id);
    if (H5Tget_class(datatype_id) == H5T_STRING)
    {
        if (H5Tis_variable_str(datatype_id))
        {
            H5Aread(attr_id, datatype_id, &value);
        }
        else
        {
            const hid_t native_type = H5Tget_native_type(
                    datatype_id, H5T_DIR_DEFAULT
            );
            const size_t type_size = H5Tget_size(native_type);
            H5Tclose(native_type);
            value = (char*) calloc(1 + len, type_size);
            H5Aread(attr_id, datatype_id, value);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        oskar_log_error(
                0, "HDF5 attribute '%s' on '%s' is not a string",
                attribute_name, object_path
        );
    }

    /* Close/release resources. */
    H5Aclose(attr_id);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) object_path;
    (void) attribute_name;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
    return value;
}


void oskar_hdf5_read_attributes(
        const oskar_HDF5* handle,
        const char* object_path,
        int* num_attributes,
        oskar_Mem*** names,
        oskar_Mem*** values,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    if (!num_attributes || !names || !values)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    oskar_mutex_lock(handle->mutex);

    /* Open the object. */
    herr_t hdf5_error = 0;
    const hid_t obj_id = H5Oopen(
            handle->file_id,
            (object_path && strlen(object_path) > 0) ? object_path : "/",
            H5P_DEFAULT
    );
    if (obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);                /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error opening HDF5 object '%s'.", object_path
        );
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Clear any old attributes. */
    for (int i = 0; i < *num_attributes; ++i)
    {
        oskar_mem_free((*names)[i], status);
        oskar_mem_free((*values)[i], status);
    }

    /* Get attributes. */
    *num_attributes = H5Aget_num_attrs(obj_id);
    const size_t sz = *num_attributes * sizeof(oskar_Mem*);
    *names = (oskar_Mem**) realloc(*names, sz);
    *values = (oskar_Mem**) realloc(*values, sz);
    for (int i = 0; i < *num_attributes; ++i)
    {
        (*names)[i] = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
        (*values)[i] = 0;
    }
    for (int i = 0; i < *num_attributes; ++i)
    {
        /* Get the attribute name. */
        const hid_t attr_id = H5Aopen_idx(obj_id, i);
        const ssize_t name_len = 1 + H5Aget_name(attr_id, 0, 0);
        oskar_mem_realloc((*names)[i], name_len, status);
        (void) H5Aget_name(attr_id, name_len, oskar_mem_char((*names)[i]));

        /* Get the attribute type, dimensions and value. */
        const hid_t dataspace_id = H5Aget_space(attr_id);
        const hid_t datatype_id = H5Aget_type(attr_id);
        const size_t num_elements =
                (size_t) H5Sget_simple_extent_npoints(dataspace_id);
        switch (H5Tget_class(datatype_id))
        {
        case H5T_INTEGER:
        {
            (*values)[i] = oskar_mem_create(
                    OSKAR_INT, OSKAR_CPU, num_elements, status
            );
            if (!*status)
            {
                hdf5_error = H5Aread(
                        attr_id, H5T_NATIVE_INT, oskar_mem_void((*values)[i])
                );
            }
            break;
        }
        case H5T_FLOAT:
        {
            (*values)[i] = oskar_mem_create(
                    OSKAR_DOUBLE, OSKAR_CPU, num_elements, status
            );
            if (!*status)
            {
                hdf5_error = H5Aread(
                        attr_id, H5T_NATIVE_DOUBLE,
                        oskar_mem_void((*values)[i])
                );
            }
            break;
        }
        case H5T_STRING:
        {
            if (H5Tis_variable_str(datatype_id))
            {
                char* data = 0;
                hdf5_error = H5Aread(attr_id, datatype_id, &data);
                if (hdf5_error >= 0)
                {
                    const size_t buffer_size = 1 + strlen(data);
                    (*values)[i] = oskar_mem_create(
                            OSKAR_CHAR, OSKAR_CPU, buffer_size, status
                    );
                    memcpy(oskar_mem_char((*values)[i]), data, buffer_size);
                }
                H5free_memory(data);
            }
            else
            {
                const hid_t native_type = H5Tget_native_type(
                        datatype_id, H5T_DIR_DEFAULT
                );
                const size_t type_size = H5Tget_size(native_type);
                H5Tclose(native_type);
                (*values)[i] = oskar_mem_create(
                        OSKAR_CHAR, OSKAR_CPU,
                        1 + num_elements * type_size, status
                );
                if (!*status)
                {
                    hdf5_error = H5Aread(
                            attr_id, datatype_id, oskar_mem_void((*values)[i])
                    );
                }
            }
            break;
        }
        default:                                          /* LCOV_EXCL_LINE */
            break; /* Ignore unknown attributes. */       /* LCOV_EXCL_LINE */
        }
        H5Tclose(datatype_id);
        H5Sclose(dataspace_id);
        H5Aclose(attr_id);
    }

    /* Close/release resources. */
    H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);

    /* Check for errors. */
    if (!*status && hdf5_error < 0)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error reading HDF5 attributes (code %d).", hdf5_error
        );
    }
#else
    (void) handle;
    (void) object_path;
    (void) num_attributes;
    (void) names;
    (void) values;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}


void oskar_hdf5_read_dataset_dims(
        const oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int* num_dims,
        size_t** dims,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    if (!dataset_name || strlen(dataset_name) == 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    oskar_mutex_lock(handle->mutex);

    /* Get the parent object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, parent_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 parent to read dimensions "
                "from dataset '%s' in '%s'.",
                dataset_name, parent_path
        );
        return;
    }

    /* Open the dataset. */
    const hid_t dataset_id = H5Dopen(obj_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);                /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error opening HDF5 dataset '%s'.", dataset_name
        );
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Read dataset dimensions. */
    read_dims(handle, dataset_id, num_dims, dims, status);

    /* Close/release resources. */
    H5Dclose(dataset_id);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) parent_path;
    (void) dataset_name;
    (void) num_dims;
    (void) dims;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}


oskar_Mem* oskar_hdf5_read_dataset(
        const oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int* num_dims,
        size_t** dims,
        int* status
)
{
    oskar_Mem* data = 0;
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return 0;
    if (!dataset_name || strlen(dataset_name) == 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return data;
    }
    oskar_mutex_lock(handle->mutex);

    /* Get the parent object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, parent_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 parent to read dataset '%s' in '%s'.",
                dataset_name, parent_path
        );
        return data;
    }

    /* Open the dataset. */
    const hid_t dataset_id = H5Dopen(obj_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);                /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error opening HDF5 dataset '%s'.", dataset_name
        );
        return data;                                      /* LCOV_EXCL_LINE */
    }

    /* Read the whole dataset. */
    data = read_hyperslab(handle, dataset_id, -1, 0, 0, status);

    /* Read dataset dimensions. */
    read_dims(handle, dataset_id, num_dims, dims, status);

    /* Close/release resources. */
    H5Dclose(dataset_id);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) parent_path;
    (void) dataset_name;
    (void) num_dims;
    (void) dims;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif

    /* Return the data. */
    return data;
}


oskar_Mem* oskar_hdf5_read_hyperslab(
        const oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int num_dims,
        const size_t* offset,
        const size_t* size,
        int* status
)
{
    oskar_Mem* data = 0;
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return 0;
    if (
            !dataset_name || strlen(dataset_name) == 0 ||
            num_dims == 0 || !offset || !size
    )
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return data;
    }
    oskar_mutex_lock(handle->mutex);

    /* Get the parent object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, parent_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 parent to read hyperslab "
                "from dataset '%s' in '%s'.",
                dataset_name, parent_path
        );
        return data;
    }

    /* Open the dataset. */
    const hid_t dataset_id = H5Dopen(obj_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);                /* LCOV_EXCL_LINE */
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0, "Error opening HDF5 dataset '%s'.", dataset_name
        );
        return data;                                      /* LCOV_EXCL_LINE */
    }

    /* Read hyperslab. */
    data = read_hyperslab(handle, dataset_id, num_dims, offset, size, status);

    /* Close/release resources. */
    H5Dclose(dataset_id);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) parent_path;
    (void) dataset_name;
    (void) num_dims;
    (void) offset;
    (void) size;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif

    /* Return the data. */
    return data;
}


void oskar_hdf5_write_attribute_double(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        double value,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    oskar_mutex_lock(handle->mutex);

    /* Get the object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, object_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 object for double attribute '%s' on '%s'.",
                attribute_name, object_path
        );
        return;
    }

    /* Create a dataspace for the attribute. */
    const hid_t attr_space = H5Screate(H5S_SCALAR);

    /* Create or overwrite attribute. */
    hid_t attr_id = 0;
    if (H5Aexists(obj_id, attribute_name))
    {
        attr_id = H5Aopen(obj_id, attribute_name, H5P_DEFAULT);
    }
    else
    {
        attr_id = H5Acreate(
                obj_id, attribute_name, H5T_NATIVE_DOUBLE, attr_space,
                H5P_DEFAULT, H5P_DEFAULT
        );
    }
    const herr_t hdf5_error = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &value);
    if (hdf5_error < 0)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0,
                "Error writing HDF5 double attribute '%s' on '%s' (code %d).",
                attribute_name, object_path, hdf5_error
        );
    }

    /* Close/release resources. */
    H5Aclose(attr_id);
    H5Sclose(attr_space);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) object_path;
    (void) attribute_name;
    (void) value;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}


void oskar_hdf5_write_attribute_int(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int value,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    oskar_mutex_lock(handle->mutex);

    /* Get the object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, object_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 object for integer attribute '%s' on '%s'.",
                attribute_name, object_path
        );
        return;
    }

    /* Create a dataspace for the attribute. */
    const hid_t attr_space = H5Screate(H5S_SCALAR);

    /* Create or overwrite attribute. */
    hid_t attr_id = 0;
    if (H5Aexists(obj_id, attribute_name))
    {
        attr_id = H5Aopen(obj_id, attribute_name, H5P_DEFAULT);
    }
    else
    {
        attr_id = H5Acreate(
                obj_id, attribute_name, H5T_NATIVE_INT, attr_space,
                H5P_DEFAULT, H5P_DEFAULT
        );
    }
    const herr_t hdf5_error = H5Awrite(attr_id, H5T_NATIVE_INT, &value);
    if (hdf5_error < 0)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0,
                "Error writing HDF5 integer attribute '%s' on '%s' (code %d).",
                attribute_name, object_path, hdf5_error
        );
    }

    /* Close/release resources. */
    H5Aclose(attr_id);
    H5Sclose(attr_space);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) object_path;
    (void) attribute_name;
    (void) value;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}


void oskar_hdf5_write_attribute_string(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        const char* value,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    oskar_mutex_lock(handle->mutex);

    /* Get the object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, object_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 object for string attribute '%s' on '%s'.",
                attribute_name, object_path
        );
        return;
    }

    /* Create a string datatype. */
    const hid_t type_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(type_id, 1 + strlen(value));
    H5Tset_strpad(type_id, H5T_STR_NULLTERM);

    /* Create a scalar dataspace for the attribute. */
    const hid_t attr_space = H5Screate(H5S_SCALAR);

    /* Delete the attribute if it exists (allows for longer strings). */
    if (H5Aexists(obj_id, attribute_name))
    {
        H5Adelete(obj_id, attribute_name);
    }

    /* Create the attribute. */
    const hid_t attr_id = H5Acreate(
            obj_id, attribute_name, type_id, attr_space,
            H5P_DEFAULT, H5P_DEFAULT
    );
    const herr_t hdf5_error = H5Awrite(attr_id, type_id, value);
    if (hdf5_error < 0)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        oskar_log_error(                                  /* LCOV_EXCL_LINE */
                0,
                "Error writing HDF5 string attribute '%s' on '%s' (code %d).",
                attribute_name, object_path, hdf5_error
        );
    }

    /* Close/release resources. */
    H5Aclose(attr_id);
    H5Sclose(attr_space);
    H5Tclose(type_id);
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) object_path;
    (void) attribute_name;
    (void) value;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}


void oskar_hdf5_write_dataset(
        oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int num_dims,
        const size_t* dims,
        const oskar_Mem* data,
        int create_empty,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    if (!dataset_name || strlen(dataset_name) == 0 || !data)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    oskar_mutex_lock(handle->mutex);

    /* Open or create the dataset. */
    int actual_num_dims = 0;
    hsize_t* actual_dims = 0;
    hsize_t* actual_offset = 0;
    get_data_dims(
            num_dims, 0, dims, data,
            &actual_num_dims, &actual_offset, &actual_dims
    );
    const hid_t dataset_id = open_or_create_dataset(
            handle, parent_path, dataset_name, actual_num_dims, actual_dims,
            data, status
    );

    /*
     * If required, write the data as a full hyperslab.
     * Otherwise, stop after creating the empty dataset.
     */
    if (!create_empty)
    {
        write_hyperslab(
                handle, dataset_id, actual_num_dims, actual_offset, actual_dims,
                data, status
        );
        if (*status)
        {
            oskar_log_error(
                    0, "Error writing HDF5 dataset '%s' in '%s'.",
                    dataset_name, parent_path
            );
        }
    }

    /* Close/release resources. */
    free(actual_dims);
    free(actual_offset);
    if (dataset_id > 0) H5Dclose(dataset_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) parent_path;
    (void) dataset_name;
    (void) num_dims;
    (void) dims;
    (void) data;
    (void) create_empty;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}


void oskar_hdf5_write_group(
        oskar_HDF5* handle,
        const char* parent_path,
        const char* group_name,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    oskar_mutex_lock(handle->mutex);

    /* Get the object ID. */
    int object_is_root = 0;
    const hid_t obj_id = get_object(
            handle, parent_path, &object_is_root, status
    );
    if (*status || obj_id < 0)
    {
        oskar_mutex_unlock(handle->mutex);
        oskar_log_error(
                0,
                "Error finding HDF5 parent to create group '%s' in '%s'.",
                group_name, parent_path
        );
        return;
    }

    /* Create the group only if it doesn't already exist. */
    if (!H5Lexists(obj_id, group_name, H5P_DEFAULT))
    {
        const hid_t group_id = H5Gcreate(
                obj_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
        );
        H5Gclose(group_id);
    }

    /* Close/release resources. */
    if (!object_is_root) H5Oclose(obj_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) parent_path;
    (void) group_name;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}


void oskar_hdf5_write_hyperslab(
        oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int num_dims,
        const size_t* offset,
        const size_t* dims,
        const oskar_Mem* data,
        int* status
)
{
#ifdef OSKAR_HAVE_HDF5
    if (*status || !handle) return;
    if (!dataset_name || 0 == strlen(dataset_name) || !data)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    oskar_mutex_lock(handle->mutex);

    /* Open or create the dataset. */
    int actual_num_dims = 0;
    hsize_t* actual_dims = 0;
    hsize_t* actual_offset = 0;
    get_data_dims(
            num_dims, offset, dims, data,
            &actual_num_dims, &actual_offset, &actual_dims
    );
    const hid_t dataset_id = open_or_create_dataset(
            handle, parent_path, dataset_name, actual_num_dims, actual_dims,
            data, status
    );

    /* Write the data to the dataset. */
    write_hyperslab(
            handle, dataset_id, actual_num_dims, actual_offset, actual_dims,
            data, status
    );
    if (*status)
    {
        oskar_log_error(
                0, "Error writing HDF5 dataset '%s' in '%s'.",
                dataset_name, parent_path
        );
    }

    /* Close/release resources. */
    free(actual_dims);
    free(actual_offset);
    if (dataset_id > 0) H5Dclose(dataset_id);
    oskar_mutex_unlock(handle->mutex);
#else
    (void) handle;
    (void) parent_path;
    (void) dataset_name;
    (void) num_dims;
    (void) offset;
    (void) dims;
    (void) data;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HDF5 support.");
#endif
}

#ifdef __cplusplus
}
#endif
