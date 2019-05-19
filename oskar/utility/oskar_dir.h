/*
 * Copyright (c) 2016-2019, The University of Oxford
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

#ifndef OSKAR_DIR_H_
#define OSKAR_DIR_H_

/**
 * @file oskar_dir.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the application's current working directory.
 *
 * @details
 * This function returns the application's current working directory.
 *
 * The returned string must be deallocated using free() when no longer needed.
 *
 * @return Returns a string containing the current working directory.
 */
OSKAR_EXPORT
char* oskar_dir_cwd(void);


/**
 * @brief Checks if the specified directory exists.
 *
 * @details
 * This function returns true if the specified directory exists, false if not.
 *
 * @param[in] dir_path  Path of the directory to check.
 *
 * @return Returns true if the specified directory exists, false if not.
 */
OSKAR_EXPORT
int oskar_dir_exists(const char* dir_path);


/**
 * @brief Checks if the specified file exists in the specified directory.
 *
 * @details
 * This function returns true if the specified file exists in the
 * specified directory, false if not.
 *
 * @param[in] dir_path  Path of the directory to check.
 * @param[in] file_name Name of the file in the directory.
 *
 * @return Returns true if the file exists in the directory, false if not.
 */
OSKAR_EXPORT
int oskar_dir_file_exists(const char* dir_path, const char* file_name);


/**
 * @brief Returns the path name of an item in the user's home directory.
 *
 * @details
 * Returns the path name of a file in the user's home directory.
 *
 * The returned string must be deallocated using free() when no longer needed.
 *
 * @param[in] item_name Name of the item in the directory.
 * @param[out] exists   If not NULL, true if the item exists, false if not.
 */
OSKAR_EXPORT
char* oskar_dir_get_home_path(const char* item_name, int* exists);


/**
 * @brief Returns the path name of an item in the directory.
 *
 * @details
 * Returns the path name of a file in the directory.
 * Does not check if the file actually exists in the current directory.
 *
 * The returned string must be deallocated using free() when no longer needed.
 *
 * @param[in] dir_path  Path of the directory.
 * @param[in] item_name Name of the item in the directory.
 */
OSKAR_EXPORT
char* oskar_dir_get_path(const char* dir_path, const char* item_name);


/**
 * @brief Returns the user's home directory.
 *
 * @details
 * This function returns the user's home directory.
 *
 * The returned string must be deallocated using free() when no longer needed.
 *
 * @return Returns a string containing the user's home directory.
 */
OSKAR_EXPORT
char* oskar_dir_home(void);


/**
 * @brief Returns names of all items in the specified directory.
 *
 * @details
 * This function returns the names of all items in the specified directory,
 * sorted by name.
 *
 * The list of names and the names themselves must be freed using free()
 * when no longer needed.
 *
 * An optional wildcard pattern can be supplied if required.
 * If not NULL, this is used to select item names matching only this pattern
 * (e.g. "*.txt" or "element_pattern*").
 *
 * @param[in] dir_path       Path of the directory.
 * @param[in] wildcard       Optional wildcard string for name match (or NULL).
 * @param[in] match_files    If true, match files.
 * @param[in] match_dirs     If true, match directories.
 * @param[in,out] num_items  Number of items in the list of names.
 * @param[in,out] items      Pointer to list of item names.
 */
OSKAR_EXPORT
void oskar_dir_items(const char* dir_path, const char* wildcard,
        int match_files, int match_dirs, int* num_items, char*** items);


/**
 * @brief Returns leafname of item from a pathname.
 *
 * @details
 * This function returns the leafname (the item name) from a pathname,
 * i.e. a pointer to the part of the string after the last directory separator.
 *
 * @param[in] path  Pathname of the item.
 *
 * @return Returns a pointer to the item's leafname.
 */
OSKAR_EXPORT
const char* oskar_dir_leafname(const char* path);


/**
 * @brief Creates a directory.
 *
 * @details
 * This function creates a directory with the given name.
 * It does nothing and returns true if the directory already exists.
 *
 * @param[in] dir_path  Path of the directory to create.
 *
 * @return Returns true if the directory was created successfully, false if not.
 */
OSKAR_EXPORT
int oskar_dir_mkdir(const char* dir_path);


/**
 * @brief Creates a directory tree.
 *
 * @details
 * This function creates a directory tree with the given path.
 * It does nothing and returns true if the directory tree already exists.
 *
 * @param[in] dir_path  Path of the directory tree to create.
 *
 * @return Returns true if the directory was created successfully, false if not.
 */
OSKAR_EXPORT
int oskar_dir_mkpath(const char* dir_path);


/**
 * @brief Removes a directory and its contents.
 *
 * @details
 * This function recursively removes the named directory and its contents.
 * Use with caution!
 *
 * @param[in] dir_path  Path of the directory to remove.
 *
 * @return Returns true if the directory was removed successfully, false if not.
 */
OSKAR_EXPORT
int oskar_dir_remove(const char* dir_path);


/**
 * @brief Returns the character denoting the directory separator.
 *
 * @details
 * This function returns the character used to separate directories in a
 * pathname: '/' on POSIX-compliant systems, '\' on Windows.
 */
OSKAR_EXPORT
char oskar_dir_separator(void);


#ifdef __cplusplus
}
#endif

#endif /* include guard */
