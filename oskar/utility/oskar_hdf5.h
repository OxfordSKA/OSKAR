/*
 * Copyright (c) 2020-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_HDF5_H_
#define OSKAR_HDF5_H_

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OSKAR_HDF5_TYPEDEF_
#define OSKAR_HDF5_TYPEDEF_
typedef struct oskar_HDF5 oskar_HDF5;
#endif

/**
 * @brief Opens a HDF5 file.
 *
 * @details
 * Opens a HDF5 file.
 *
 * @param[in] file_path  Pathname to HDF5 file.
 * @param[in] mode       Either 'r', 'w' or 'a' for read/write/append.
 * @param[in,out] status Status return code.
 *
 * @return A handle to the opened file.
 */
OSKAR_EXPORT
oskar_HDF5* oskar_hdf5_open(const char* file_path, char mode, int* status);

/**
 * @brief Decrements the reference count, freeing resources as needed.
 *
 * @details
 * Decrements the reference count, freeing resources as needed.
 *
 * @param[in] handle  Handle to HDF5 file.
 */
OSKAR_EXPORT
void oskar_hdf5_close(oskar_HDF5* handle);

/**
 * @brief Increments the reference count.
 *
 * @details
 * Increments the reference count.
 *
 * @param[in] handle  Handle to HDF5 file.
 */
OSKAR_EXPORT
void oskar_hdf5_ref_inc(oskar_HDF5* handle);

/**
 * @brief Returns true if the dataset exists.
 *
 * @param[in] handle  Handle to HDF5 file.
 * @param[in] name    The name (path) of a dataset in the file.
 */
OSKAR_EXPORT
int oskar_hdf5_dataset_exists(const oskar_HDF5* handle, const char* name);

/**
 * @brief Returns the name (path) of a dataset in the file.
 *
 * @param[in] handle  Handle to HDF5 file.
 * @param[in] i       Index of dataset.
 */
OSKAR_EXPORT
const char* oskar_hdf5_dataset_name(const oskar_HDF5* handle, int i);

/**
 * @brief Returns the number of datasets in the HDF5 file.
 *
 * @param[in] handle  Handle to HDF5 file.
 */
OSKAR_EXPORT
int oskar_hdf5_num_datasets(const oskar_HDF5* handle);

/**
 * @brief Reads a named attribute of an object as a single integer.
 *
 * @param[in] handle         Handle to HDF5 file.
 * @param[in] object_path    The path of an object in the file.
 * @param[in] attribute_name The name of the attribute to read.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
int oskar_hdf5_read_attribute_int(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int* status
);

/**
 * @brief Reads a named attribute of an object as a single double.
 *
 * @param[in] handle         Handle to HDF5 file.
 * @param[in] object_path    The path of an object in the file.
 * @param[in] attribute_name The name of the attribute to read.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
double oskar_hdf5_read_attribute_double(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int* status
);

/**
 * @brief Reads a named attribute of an object as a string.
 *
 * @param[in] handle         Handle to HDF5 file.
 * @param[in] object_path    The path of an object in the file.
 * @param[in] attribute_name The name of the attribute to read.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
char* oskar_hdf5_read_attribute_string(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int* status
);

/**
 * @brief Reads attributes associated with an object in the HDF5 file.
 *
 * @details
 * Reads attributes associated with an object in the HDF5 file.
 *
 * @param[in] handle             Handle to HDF5 file.
 * @param[in] object_path        The path of an object in the file.
 * @param[in,out] num_attributes The size of the attribute arrays.
 * @param[in,out] names          The name of each attribute.
 * @param[in,out] values         The value of each attribute.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_read_attributes(
        const oskar_HDF5* handle,
        const char* object_path,
        int* num_attributes,
        oskar_Mem*** names,
        oskar_Mem*** values,
        int* status
);

/**
 * @brief Reads the dimensions of a dataset.
 *
 * @details
 * Reads the dimensions of a dataset.
 *
 * @param[in] handle       Handle to HDF5 file.
 * @param[in] parent_path  The path of the parent group, which must exist.
 * @param[in] dataset_name The name of the dataset to read.
 * @param[out] num_dims    The number of dimensions in the dataset.
 * @param[in,out] dims     The size of each dimension in the dataset.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_read_dataset_dims(
        const oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int* num_dims,
        size_t** dims,
        int* status
);

/**
 * @brief Reads a whole dataset from the HDF5 file.
 *
 * @details
 * Reads a whole dataset from the HDF5 file.
 *
 * @param[in] handle       Handle to HDF5 file.
 * @param[in] parent_path  The path of the parent group, which must exist.
 * @param[in] dataset_name The name of the dataset to read.
 * @param[out] num_dims    The number of dimensions in the dataset.
 * @param[in,out] dims     The size of each dimension in the dataset.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_hdf5_read_dataset(
        const oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int* num_dims,
        size_t** dims,
        int* status
);

/**
 * @brief Reads a part of a dataset from the HDF5 file.
 *
 * @details
 * Reads a part of a dataset from the HDF5 file.
 *
 * @param[in] handle       Handle to HDF5 file.
 * @param[in] parent_path  The path of the parent group, which must exist.
 * @param[in] dataset_name The name of the dataset to read.
 * @param[in] num_dims     The number of dimensions to read.
 * @param[in] offset       The start offset of each dimension.
 * @param[in] size         The number of elements of each dimension to read.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_hdf5_read_hyperslab(
        const oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int num_dims,
        const size_t* offset,
        const size_t* size,
        int* status
);

/**
 * @brief Writes a single double as a named attribute of an object.
 *
 * @param[in] handle         Handle to HDF5 file.
 * @param[in] object_path    The path of the object. Use "/" for root.
 * @param[in] attribute_name The name of the attribute to write.
 * @param[in] value          The value of the attribute to write.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_write_attribute_double(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        double value,
        int* status
);

/**
 * @brief Writes a single integer as a named attribute of an object.
 *
 * @param[in] handle         Handle to HDF5 file.
 * @param[in] object_path    The path of the object. Use "/" for root.
 * @param[in] attribute_name The name of the attribute to write.
 * @param[in] value          The value of the attribute to write.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_write_attribute_int(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        int value,
        int* status
);

/**
 * @brief Writes a string as a named attribute of an object.
 *
 * @param[in] handle         Handle to HDF5 file.
 * @param[in] object_path    The path of the object. Use "/" for root.
 * @param[in] attribute_name The name of the attribute to write.
 * @param[in] value          The value of the attribute to write.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_write_attribute_string(
        const oskar_HDF5* handle,
        const char* object_path,
        const char* attribute_name,
        const char* value,
        int* status
);

/**
 * @brief Writes a whole dataset to the HDF5 file.
 *
 * @details
 * Writes a whole dataset to the HDF5 file.
 *
 * @param[in] handle       Handle to HDF5 file.
 * @param[in] parent_path  The path of the parent group, which must exist.
 * @param[in] dataset_name The name of the new dataset.
 * @param[in] num_dims     The number of dimensions in the dataset.
 * @param[in] dims         The size of each dimension in the dataset.
 * @param[in] data         The data to write.
 * @param[in] create_empty If true, only create an empty dataset of the
 *                         given size for the given data type;
 *                         do not write any data yet.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_write_dataset(
        oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int num_dims,
        const size_t* dims,
        const oskar_Mem* data,
        int create_empty,
        int* status
);

/**
 * @brief Writes a new group to the HDF5 file.
 *
 * @details
 * Writes a new group to the HDF5 file.
 *
 * @param[in] handle      Handle to HDF5 file.
 * @param[in] parent_path The path of the parent group.
 * @param[in] group_name  The name of the new group to create.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_write_group(
        oskar_HDF5* handle,
        const char* parent_path,
        const char* group_name,
        int* status
);

/**
 * @brief Writes part of a dataset to the HDF5 file.
 *
 * @details
 * Writes part of a dataset to the HDF5 file.
 *
 * @param[in] handle       Handle to HDF5 file.
 * @param[in] parent_path  The path of the parent group, which must exist.
 * @param[in] dataset_name The name of the new dataset.
 * @param[in] num_dims     The number of dimensions to write.
 * @param[in] offset       The start offset of each dimension.
 * @param[in] dims         The number of elements of each dimension to write.
 * @param[in] data         The data to write.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_write_hyperslab(
        oskar_HDF5* handle,
        const char* parent_path,
        const char* dataset_name,
        int num_dims,
        const size_t* offset,
        const size_t* dims,
        const oskar_Mem* data,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
