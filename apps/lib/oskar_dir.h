/*
 * Copyright (c) 2013-2015, The University of Oxford
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
 * @brief Removes a directory and its contents.
 *
 * @details
 * This function recursively removes the named directory and its contents.
 *
 * @param[in] dir_name Name of the directory to remove.
 */
OSKAR_APPS_EXPORT
int oskar_dir_remove(const char* dir_name);

/**
 * @brief Checks if the specified directory exists.
 *
 * @details
 * This function returns true if the specified directory exists, false if not.
 *
 * @param[in] dir_name Name of the directory to check.
 *
 * @return Returns true if the specified directory exists, false if not
 */
OSKAR_APPS_EXPORT
int oskar_dir_exists(const char* dir_name);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

#include <string>
#include <vector>

/**
 * @brief Provides an abstraction for working with a directory hierarchy.
 *
 * @details
 * This class provides an abstraction for working with a directory hierarchy.
 * Internally, it calls third-party library functionality, currently provided
 * by the Qt framework.
 */
class OSKAR_APPS_EXPORT oskar_Dir
{
public:
    /**
     * @brief Constructs a directory object pointing to the given path.
     */
    oskar_Dir(const std::string& path);
    ~oskar_Dir();

    /**
     * @brief
     * Returns the absolute path name of a file in the directory.
     *
     * @details
     * Returns the absolute path name of a file in the directory.
     * Does not check if the file actually exists in the current directory.
     *
     * @param[in] filename Name of file.
     *
     * @return Pathname to file.
     */
    std::string absoluteFilePath(const std::string& filename) const;

    /**
     * @brief
     * Returns the absolute path of the directory.
     *
     * @details
     * Returns the absolute path of the directory.
     */
    std::string absolutePath() const;

    /**
     * @brief
     * Returns the names of all the files in the current directory.
     *
     * @details
     * Returns a list of the names of all the files in the current
     * directory. The list is sorted by name.
     */
    std::vector<std::string> allFiles() const;

    /**
     * @brief
     * Returns the names of all the directories in the current directory.
     *
     * @details
     * Returns a list of the names of all the directories in the current
     * directory. The list is sorted by name.
     */
    std::vector<std::string> allSubDirs() const;

    /**
     * @brief
     * Returns true if the specified file exists in the current directory.
     *
     * @details
     * Returns true if the specified file exists in the current directory.
     */
    bool exists(const std::string& filename) const;

    /**
     * @brief
     * Returns true if the directory exists.
     *
     * @details
     * Returns true if the directory exists.
     */
    bool exists() const;

    /**
     * @brief
     * Returns the path name of a file in the directory.
     *
     * @details
     * Returns the path name of a file in the directory.
     * Does not check if the file actually exists in the current directory.
     *
     * @param[in] filename Name of file.
     *
     * @return Pathname to file.
     */
    std::string filePath(const std::string& filename) const;

    /**
     * @brief
     * Returns the names of all the directories in the current directory.
     *
     * @details
     * Returns a list of the names of all the directories in the current
     * directory. The list is sorted by name.
     */
    std::vector<std::string> filesStartingWith(const std::string& name) const;

    /**
     * @brief
     * Removes an empty directory from the current directory.
     *
     * @details
     * Removes an empty directory from the current directory.
     *
     * @return True if successful, false if unsuccessful.
     */
    bool rmdir(const std::string& name);

    /**
     * @brief
     * Recursively remove a directory tree.
     *
     * @details
     * Recursively removes all files and directories in the specified root
     * directory, which must be given as an absolute path.
     *
     * Use with caution!
     *
     * @param[in] root Full (absolute) pathname of directory to remove.
     *
     * @return True if successful, false if unsuccessful.
     */
    static bool rmtree(const std::string& root);

private:
    struct oskar_DirPrivate;
    oskar_DirPrivate* p;
};

#endif /* __cplusplus */

#endif /* OSKAR_DIR_H_ */
