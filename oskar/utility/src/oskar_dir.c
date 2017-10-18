/*
 * Copyright (c) 2016, The University of Oxford
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

#include "utility/oskar_dir.h"

#ifndef OSKAR_OS_WIN
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_dir_exists(const char* dir_path)
{
#ifndef OSKAR_OS_WIN
    struct stat s;
    return (!stat(dir_path, &s) && S_ISDIR(s.st_mode)) ? 1 : 0;
#else
    int attrib = GetFileAttributes(dir_path);
    return (attrib != INVALID_FILE_ATTRIBUTES &&
            (attrib & FILE_ATTRIBUTE_DIRECTORY));
#endif
}


int oskar_dir_file_exists(const char* dir_path, const char* file_name)
{
    FILE* f = 0;
    char* path = 0;
    path = oskar_dir_get_path(dir_path, file_name);
    f = fopen(path, "rb");
    if (f)
    {
        fclose(f);
        free(path);
        return 1;
    }
    free(path);
    return 0;
}


char* oskar_dir_get_path(const char* dir_path, const char* item_name)
{
    char* buffer = 0;
    int buf_len = 0, dir_path_len;
    dir_path_len = (int) strlen(dir_path);
    buf_len = 2 + dir_path_len + (int) strlen(item_name);
    buffer = (char*) calloc(buf_len, sizeof(char));
    if (dir_path[dir_path_len - 1] == oskar_dir_separator())
        sprintf(buffer, "%s%s", dir_path, item_name);
    else
        sprintf(buffer, "%s%c%s", dir_path, oskar_dir_separator(), item_name);
    return buffer;
}


static int name_cmp(const void* a, const void* b)
{
    return strcmp(*(char* const*)a, *(char* const*)b);
}


/* From http://c-faq.com/lib/regex.html */
static int match(const char *pattern, const char *str)
{
    switch (*pattern)
    {
    case '\0': return !*str;
    case '*':  return match(pattern+1, str) || (*str && match(pattern, str+1));
    case '?':  return *str && match(pattern+1, str+1);
    default:   return *pattern == *str && match(pattern+1, str+1);
    }
}


static void item(const char* dir_path, const char* name, const char* wildcard,
        int match_files, int match_dirs, int* i, int num_items, char** items)
{
    if (!strcmp(name, ".") || !strcmp(name, "..")) return;
    if (wildcard && !match(wildcard, name)) return;
    if (match_files ^ match_dirs)
    {
        int is_dir = 0, rejected = 0;
        char* item_path;
        item_path = oskar_dir_get_path(dir_path, name);
        is_dir = oskar_dir_exists(item_path);
        if ((is_dir && !match_dirs) || (!is_dir && match_dirs))
            rejected = 1;
        free(item_path);
        if (rejected) return;
    }
    if (items && *i < num_items)
    {
        items[*i] = (char*) realloc(items[*i], 1 + strlen(name));
        strcpy(items[*i], name);
    }
    ++(*i);
}


static int get_items(const char* dir_path, const char* wildcard,
        int match_files, int match_dirs, int num_items, char** items)
{
    int i = 0;
#ifndef OSKAR_OS_WIN
    DIR *d;
    struct dirent *t;
    if (!(d = opendir(dir_path))) return 0;
    while ((t = readdir(d)) != 0)
        item(dir_path, t->d_name, wildcard, match_files, match_dirs,
                &i, num_items, items);
    (void) closedir(d);
#else
    WIN32_FIND_DATA f;
    HANDLE h = 0;
    char* buffer = 0;
    buffer = (char*) malloc(3 + strlen(dir_path));
    (void) sprintf(buffer, "%s\\*", dir_path);
    if ((h = FindFirstFile(buffer, &f)) == INVALID_HANDLE_VALUE)
    {
        free(buffer);
        return 0;
    }
    do
        item(dir_path, f.cFileName, wildcard, match_files, match_dirs,
                &i, num_items, items);
    while (FindNextFile(h, &f));
    FindClose(h);
    free(buffer);
#endif
    return i;
}


void oskar_dir_items(const char* dir_path, const char* wildcard,
        int match_files, int match_dirs, int* num_items, char*** items)
{
    int i = 0, old_num_items = *num_items;

    /* Count the number of items. */
    *num_items = get_items(dir_path, wildcard, match_files, match_dirs, 0, 0);

    /* Get the sorted list of names if required. */
    if (items)
    {
        for (i = *num_items; i < old_num_items; ++i) free((*items)[i]);
        *items = (char**) realloc(*items, *num_items * sizeof(char**));
        for (i = old_num_items; i < *num_items; ++i) (*items)[i] = 0;
        (void) get_items(dir_path, wildcard, match_files, match_dirs,
                *num_items, *items);
        qsort(*items, *num_items, sizeof(char*), name_cmp);
    }
}


int oskar_dir_mkdir(const char* dir_path)
{
#ifndef OSKAR_OS_WIN
    struct stat s;
    if (stat(dir_path, &s) != 0)
    {
        /* Item does not exist. Try to create the directory. */
        if (mkdir(dir_path, 0755) != 0 && errno != EEXIST)
            return 0;
    }
    else if (!S_ISDIR(s.st_mode))
        return 0;
#else
    int attrib;
    if ((attrib = GetFileAttributes(dir_path)) == INVALID_FILE_ATTRIBUTES)
    {
        /* Item does not exist. Try to create the directory. */
        if (!CreateDirectory(dir_path, 0) &&
                GetLastError() != ERROR_ALREADY_EXISTS)
            return 0;
    }
    else if (!(attrib & FILE_ATTRIBUTE_DIRECTORY))
        return 0;
#endif
    return 1;
}


int oskar_dir_mkpath(const char* dir_path)
{
    char *start, *sep, *dir_path_p;
    int error = 0;
    size_t path_len;

    /* Copy the input path string so it can be modified. */
    path_len = 1 + strlen(dir_path);
    dir_path_p = malloc(path_len);
    if (!dir_path_p) return 0;
    memcpy(dir_path_p, dir_path, path_len);

    /* Loop over directories in path to ensure they all exist. */
    start = dir_path_p;
    while (!error && (sep = strchr(start, oskar_dir_separator())))
    {
        if (sep != start && *(sep - 1) != ':')
        {
            *sep = '\0'; /* Temporarily truncate to ensure this dir exists. */
            error = !oskar_dir_mkdir(dir_path_p);
            *sep = oskar_dir_separator(); /* Restore full path. */
        }
        start = sep + 1;
    }
    free(dir_path_p);
    return !error ? oskar_dir_mkdir(dir_path) : 0;
}


int oskar_dir_remove(const char* dir_path)
{
    int error = 0, i, num_items = 0;
    char **items = 0, *path = 0;
    if (!oskar_dir_exists(dir_path) ||
            !strcmp(dir_path, ".") ||
            !strcmp(dir_path, "./") ||
            !strcmp(dir_path, ".\\"))
        return 0;

    /* Get names of all items in the directory. */
    oskar_dir_items(dir_path, NULL, 1, 1, &num_items, &items);
    for (i = 0; i < num_items; ++i)
    {
        /* Get full path of the item. */
        path = oskar_dir_get_path(dir_path, items[i]);

        /* Remove files and directories recursively. */
        if (!oskar_dir_exists(path))
            error = remove(path);
        else
            error = !oskar_dir_remove(path);
        free(path);
        if (error) break;
    }
    for (i = 0; i < num_items; ++i) free(items[i]);
    free(items);
    if (error) return 0;

    /* Remove the empty directory. */
#ifdef OSKAR_OS_WIN
    return RemoveDirectory(dir_path);
#else
    return remove(dir_path) ? 0 : 1;
#endif
}


char oskar_dir_separator(void)
{
#ifdef OSKAR_OS_WIN
    return '\\';
#else
    return '/';
#endif
}


#ifdef __cplusplus
}
#endif
