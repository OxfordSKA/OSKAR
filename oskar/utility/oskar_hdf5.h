/*
 * Copyright (c) 2020, The OSKAR Developers.
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
 * @brief Opens a HDF5 file for reading.
 *
 * @details
 * Opens a HDF5 file for reading.
 *
 * @param[in] file_path  Pathname to HDF5 file.
 * @param[in,out] status Status return code.
 *
 * @return A handle to the opened file.
 */
OSKAR_EXPORT
oskar_HDF5* oskar_hdf5_open(const char* file_path, int* status);

/**
 * @brief Closes the HDF5 file.
 *
 * @details
 * Closes the HDF5 file.
 *
 * @param[in] h  Handle to HDF5 file.
 */
OSKAR_EXPORT
void oskar_hdf5_close(oskar_HDF5* h);

/**
 * @brief Returns the name (path) of a dataset in the file.
 *
 * @details
 * Returns the name (path) of a dataset in the file.
 *
 * @param[in] h  Handle to HDF5 file.
 * @param[in] i  Index of dataset.
 */
OSKAR_EXPORT
const char* oskar_hdf5_dataset_name(const oskar_HDF5* h, int i);

/**
 * @brief Returns the number of datasets in the HDF5 file.
 *
 * @details
 * Returns the number of datasets in the HDF5 file.
 *
 * @param[in] h  Handle to HDF5 file.
 */
OSKAR_EXPORT
int oskar_hdf5_num_datasets(const oskar_HDF5* h);

/**
 * @brief Reads the dimensions of a dataset.
 *
 * @details
 * Reads the dimensions of a dataset.
 *
 * @param[in] h  Handle to HDF5 file.
 * @param[in] dataset_path  The name (path) of a dataset in the file.
 * @param[out] num_dims  The number of dimensions in the dataset.
 * @param[in,out] dims   The size of each dimension in the dataset.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_hdf5_read_dataset_dims(oskar_HDF5* h, const char* dataset_path,
        int* num_dims, size_t** dims, int* status);

/**
 * @brief Reads a whole dataset from the HDF5 file.
 *
 * @details
 * Reads a whole dataset from the HDF5 file.
 *
 * @param[in] h  Handle to HDF5 file.
 * @param[in] dataset_path  The name (path) of a dataset in the file.
 * @param[out] num_dims  The number of dimensions in the dataset.
 * @param[in,out] dims   The size of each dimension in the dataset.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_hdf5_read_dataset(oskar_HDF5* h, const char* dataset_path,
        int* num_dims, size_t** dims, int* status);

/**
 * @brief Reads a part of a dataset from the HDF5 file.
 *
 * @details
 * Reads a part of a dataset from the HDF5 file.
 *
 * @param[in] h  Handle to HDF5 file.
 * @param[in] dataset_path  The name (path) of a dataset in the file.
 * @param[in] num_dims   The number of dimensions to read.
 * @param[in] offset     The start offset of each dimension.
 * @param[in] size       The number of elements of each dimension to read.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_hdf5_read_hyperslab(oskar_HDF5* h, const char* dataset_path,
        int num_dims, const size_t* offset, const size_t* size, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
