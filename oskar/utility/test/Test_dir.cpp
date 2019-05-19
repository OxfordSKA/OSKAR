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

#include <gtest/gtest.h>
#include "utility/oskar_dir.h"
#include <cstdio>


TEST(dir, create)
{
    const char* top = "test1";
    char path[32];
    char sep = oskar_dir_separator();
    sprintf(path, "%s%ctest2%ctest3", top, sep, sep);
    ASSERT_TRUE(oskar_dir_mkpath(path));
    ASSERT_TRUE(oskar_dir_remove(top));
}


TEST(dir, list)
{
    int n = 0;
    char** d = 0;

    oskar_dir_items(".", NULL, true, true, &n, &d);
//    for (int i = 0; i < n; ++i) printf("%15s: %s\n", "All items", d[i]);
    oskar_dir_items(".", NULL, true, false, &n, &d);
//    for (int i = 0; i < n; ++i) printf("%15s: %s\n", "File", d[i]);
    oskar_dir_items(".", "oskar*", true, false, &n, &d);
//    for (int i = 0; i < n; ++i) printf("%15s: %s\n", "File selection", d[i]);
    oskar_dir_items(".", NULL, false, true, &n, &d);
//    for (int i = 0; i < n; ++i) printf("%15s: %s\n", "Dir", d[i]);
    oskar_dir_items(".", "s*", false, true, &n, &d);
//    for (int i = 0; i < n; ++i) printf("%15s: %s\n", "Dir selection", d[i]);

    for (int i = 0; i < n; ++i) free(d[i]);
    free(d);
}


TEST(dir, home)
{
    int exists = 0;
    char* home_path = oskar_dir_get_home_path("foo.txt", &exists);
//    printf("Path to item: %s, Exists: %i\n", home_path, exists);
    free(home_path);
}
