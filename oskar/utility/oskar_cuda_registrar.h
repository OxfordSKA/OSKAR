/*
 * Copyright (c) 2018-2019, The University of Oxford
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

#ifndef OSKAR_CUDA_REGISTRAR_H_
#define OSKAR_CUDA_REGISTRAR_H_

#include <stdlib.h>

namespace oskar {

/* Note: Using *ANY* STL container here causes multiple link-time failures
 * (nvcc probably uses a different version of libc++), so use custom
 * containers here instead, and construct the kernel map on first use.
 * We only need size(), push_back() and operator[] anyway... */

struct CudaKernelRegistrar
{
    struct Pair {
        const char* first;
        const void* second;
        Pair(const char* name, const void* ptr) : first(name), second(ptr) {}
    };
    class List {
        Pair* list_;
        int size_;
    public:
        List() : list_(0), size_(0) {}
        virtual ~List() { free(list_); size_ = 0; }
        int size() const { return size_; }
        void push_back(const Pair& value)
        {
            size_++;
            list_ = (Pair*) realloc(list_, size_ * sizeof(Pair));
            list_[size_-1].first = value.first;
            list_[size_-1].second = value.second;
        }
        const Pair& operator[](int i) const { return list_[i]; }
    };
    static List& kernels() { static List k; return k; }
    CudaKernelRegistrar(const char* name, const void* ptr)
    {
        kernels().push_back(Pair(name, ptr));
    }
};

}

#define M_CAT(A, B) M_CAT_(A, B)
#define M_CAT_(A, B) A##B

#define OSKAR_CUDA_KERNEL(NAME) \
    static oskar::CudaKernelRegistrar M_CAT(r_, NAME)(#NAME, (const void*) &NAME);

#endif /* include guard */
