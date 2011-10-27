/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_MEX_POINTER_H_
#define OSKAR_MEX_POINTER_H_

/**
 * @file oskar_mex_pointer.h
 */

#ifdef _MSC_VER
    typedef __int32 int32_t;
    typedef unsigned __int32 uint32_t;
    typedef unsigned __int64 uint64_t;
#else
    #include <stdint.h>
#endif

#include <mex.h>

// NOTE using a slightly more sophisticated handle class here with a
// magic number signature might be a good idea here.
// To see how to do this see the mex function class testing code

/**
 * @brief Converts a pointer of a specified type to an mxArray containing the
 * pointer.
 *
 * @param[in] ptr Pointer to the object to convert to an mxArray.
 *
 * @return mxArray pointer holding the type pointer.
 */
template <class T> inline mxArray* convert_pointer_to_mxArray(T* ptr)
{
    mxArray* out = mxCreateNumericMatrix(1,1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(ptr);
    return out;
}

/**
 * @brief Converts a pointer stored as an mxArray to a pointer of the
 * specified type.
 *
 * @param[in] in mxArray pointer holding the value of the pointer to the object.
 *
 * @return Pointer to the specified type.
 */
template<class T> inline T* covert_mxArray_to_pointer(const mxArray* in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS
            || mxIsComplex(in))
    {
        mexErrMsgTxt("ERROR: Input must be a real uint64 scalar.");
    }
    return reinterpret_cast<T*>(*((uint64_t *)mxGetData(in)));
}

#endif // OSKAR_MEX_POINTER_H_
