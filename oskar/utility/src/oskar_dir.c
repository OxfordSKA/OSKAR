/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_dir.h"

#ifndef OSKAR_OS_WIN
#define _BSD_SOURCE
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
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

char* oskar_dir_cwd(void)
{
    char* str = 0;
    size_t len = 0;
#ifndef OSKAR_OS_WIN
    char* new_str = 0;
    do
    {
        len += 256;
        new_str = (char*) realloc(str, len);
        if (!new_str)
        {
            free(str);
            return 0;
        }
        str = new_str;
    }
    while (getcwd(str, len) == NULL);
#else
    len = GetCurrentDirectory(0, NULL);
    str = (char*) calloc(len, sizeof(char));
    if (!str) return 0;
    GetCurrentDirectory((DWORD) len, str);
#endif
    return str;
}


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


char* oskar_dir_get_home_path(const char* item_name, int* exists)
{
    char *path = 0, *home_dir = 0;
    home_dir = oskar_dir_home();
    path = oskar_dir_get_path(home_dir, item_name);
    if (exists) *exists = oskar_dir_file_exists(home_dir, item_name);
    free(home_dir);
    return path;
}


char* oskar_dir_get_path(const char* dir_path, const char* item_name)
{
    char* buffer = 0;
    const size_t dir_path_len = strlen(dir_path);
    const size_t buf_len = 2 + dir_path_len + strlen(item_name);
    buffer = (char*) calloc(buf_len, sizeof(char));
    if (!buffer) return 0;
    if (dir_path[dir_path_len - 1] == oskar_dir_separator())
    {
        sprintf(buffer, "%s%s", dir_path, item_name);
    }
    else
    {
        sprintf(buffer, "%s%c%s", dir_path, oskar_dir_separator(), item_name);
    }
    return buffer;
}


char* oskar_dir_home(void)
{
    char *tmp = 0, *home_dir = 0;
#ifndef OSKAR_OS_WIN
    char *buffer = 0;
    struct passwd pwd, *result = 0;
    tmp = getenv("HOME");
    if (!tmp)
    {
        long int buf_len = sysconf(_SC_GETPW_R_SIZE_MAX);
        if (buf_len == -1) buf_len = 16384;
        buffer = (char*) malloc((size_t) buf_len);
        if (!buffer) return 0;
        getpwuid_r(geteuid(), &pwd, buffer, (size_t) buf_len, &result);
        if (result != NULL) tmp = pwd.pw_dir;
    }
    if (tmp)
    {
        const size_t buffer_size = 1 + strlen(tmp);
        home_dir = (char*) calloc(buffer_size, 1);
        if (home_dir) memcpy(home_dir, tmp, buffer_size);
    }
    free(buffer);
#else
    tmp = getenv("USERPROFILE");
    if (tmp)
    {
        const size_t buffer_size = 1 + strlen(tmp);
        home_dir = (char*) calloc(buffer_size, 1);
        if (home_dir) memcpy(home_dir, tmp, buffer_size);
    }
#endif
    return home_dir;
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
        int rejected = 0;
        char* item_path = 0;
        item_path = oskar_dir_get_path(dir_path, name);
        const int is_dir = oskar_dir_exists(item_path);
        if ((is_dir && !match_dirs) || (!is_dir && match_dirs)) rejected = 1;
        free(item_path);
        if (rejected) return;
    }
    if (items && *i < num_items)
    {
        char* new_item = 0;
        const size_t buffer_size = 1 + strlen(name);
        new_item = (char*) realloc(items[*i], buffer_size);
        if (new_item)
        {
            items[*i] = new_item;
            memcpy(items[*i], name, buffer_size);
        }
    }
    ++(*i);
}


static int get_items(const char* dir_path, const char* wildcard,
        int match_files, int match_dirs, int num_items, char** items)
{
    int i = 0;
#ifndef OSKAR_OS_WIN
    DIR *d = 0;
    struct dirent *t = 0;
    if (!(d = opendir(dir_path))) return 0;
    while ((t = readdir(d)) != 0)
    {
        item(dir_path, t->d_name, wildcard, match_files, match_dirs,
                &i, num_items, items);
    }
    (void) closedir(d);
#else
    WIN32_FIND_DATA f;
    HANDLE h = 0;
    char* buffer = 0;
    buffer = (char*) malloc(3 + strlen(dir_path));
    if (!buffer) return 0;
    (void) sprintf(buffer, "%s\\*", dir_path);
    if ((h = FindFirstFile(buffer, &f)) == INVALID_HANDLE_VALUE)
    {
        free(buffer);
        return 0;
    }
    do
    {
        item(dir_path, f.cFileName, wildcard, match_files, match_dirs,
                &i, num_items, items);
    }
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
    if (!items) return;
    if (*num_items == 0)
    {
        /* If the new list is empty, free the old one and return. */
        goto cleanup;
    }
    else
    {
        char** new_items = 0;

        /* If the old list is larger, free the extra items. */
        for (i = *num_items; i < old_num_items; ++i) free((*items)[i]);

        /* Resize the list and check for errors. */
        new_items = (char**) realloc(*items, *num_items * sizeof(char*));
        if (!new_items) goto cleanup;
        *items = new_items;

        /* If the new list is larger, initialise the extra items. */
        for (i = old_num_items; i < *num_items; ++i) (*items)[i] = 0;

        /* Get the items and sort them. */
        (void) get_items(dir_path, wildcard, match_files, match_dirs,
                *num_items, *items);
        qsort(*items, *num_items, sizeof(char*), name_cmp);
    }
    return;

cleanup:
    for (i = 0; i < old_num_items; ++i) free((*items)[i]);
    free(*items);
    *items = 0;
    *num_items = 0;
}


const char* oskar_dir_leafname(const char* path)
{
    const char* leafname = strrchr(path, (int)oskar_dir_separator());
    return !leafname ? path : leafname + 1;
}


int oskar_dir_mkdir(const char* dir_path)
{
#ifndef OSKAR_OS_WIN
    struct stat s;
    if (stat(dir_path, &s) != 0)
    {
        /* Item does not exist. Try to create the directory. */
        if (mkdir(dir_path, 0755) != 0 && errno != EEXIST) return 0;
    }
    else if (!S_ISDIR(s.st_mode))
    {
        return 0;
    }
#else
    int attrib;
    if ((attrib = GetFileAttributes(dir_path)) == INVALID_FILE_ATTRIBUTES)
    {
        /* Item does not exist. Try to create the directory. */
        if (!CreateDirectory(dir_path, 0) &&
                GetLastError() != ERROR_ALREADY_EXISTS)
        {
            return 0;
        }
    }
    else if (!(attrib & FILE_ATTRIBUTE_DIRECTORY))
    {
        return 0;
    }
#endif
    return 1;
}


int oskar_dir_mkpath(const char* dir_path)
{
    char *start = 0, *sep = 0, *dir_path_p = 0;
    int error = 0;

    /* Copy the input path string so it can be modified. */
    const size_t path_len = 1 + strlen(dir_path);
    dir_path_p = (char*) calloc(path_len, sizeof(char));
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
    int error = 0, i = 0, num_items = 0;
    char **items = 0, *path = 0;
    if (!oskar_dir_exists(dir_path) ||
            !strcmp(dir_path, ".") ||
            !strcmp(dir_path, "./") ||
            !strcmp(dir_path, ".\\"))
    {
        return 0;
    }

    /* Get names of all items in the directory. */
    oskar_dir_items(dir_path, NULL, 1, 1, &num_items, &items);
    for (i = 0; i < num_items; ++i)
    {
        /* Get full path of the item. */
        path = oskar_dir_get_path(dir_path, items[i]);

        /* Remove files and directories recursively. */
        error = !oskar_dir_exists(path) ?
                remove(path) : !oskar_dir_remove(path);
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
