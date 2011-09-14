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

#ifndef OSKAR_MATH_LINKED_SORT_H_
#define OSKAR_MATH_LINKED_SORT_H_

/**
 * @file oskar_LinkedSort.h
 */

#include <algorithm>
#include <utility>
#include <vector>
#include <cstring>


/**
 * @class oskar_LinkedSort
 * @brief
 * @details
 */
class oskar_LinkedSort
{
    public:
        oskar_LinkedSort() {}

        virtual ~oskar_LinkedSort() {}

        template <typename T0, typename T1>
        static void lSort(const unsigned n, T0 * values, T1 * linked1);

        template <typename T0, typename T1, typename T2, typename T3>
        static void lSort(const unsigned n, T0 * values, T1 * linked1,
                T2 * linked2, T3 * linked3);


        template <typename T> struct Pair
        {
                unsigned index;
                T value;
        };

    public:
        /// Sort values and return sorted values and corresponding indices.
        template <typename T>
        static void sortIndices(const unsigned n, T * values, unsigned * indices);

        /// Reorder values based on vector of indices.
        template <typename T>
        static void reorder(const unsigned n, T * values, const unsigned * indices);

    private:
        /// Comparison function for sorting indices.
        template <typename T>
        static bool _compare(const Pair<T> & l, const Pair<T>& r);
};


//-----------------------------------------------------------------------------
template <typename T0, typename T1>
void oskar_LinkedSort::lSort(const unsigned n, T0 * values, T1 * linked1)
{
    std::vector<unsigned> indices(n);
    sortIndices<T0>(n, values, &indices[0]);
    reorder<T1>(n, linked1, &indices[0]);
}


template <typename T0, typename T1, typename T2, typename T3>
void oskar_LinkedSort::lSort(const unsigned n, T0 * values, T1 * linked1,
        T2 * linked2, T3 * linked3)
{
    std::vector<unsigned> indices(n);
    sortIndices<T0>(n, values, &indices[0]);
    reorder<T1>(n, linked1, &indices[0]);
    reorder<T2>(n, linked2, &indices[0]);
    reorder<T3>(n, linked3, &indices[0]);
}

/// Sort values and return sorted values and corresponding indices.
template <typename T>
void oskar_LinkedSort::sortIndices(const unsigned n, T * values, unsigned * indices)
{
    std::vector<Pair<T> > temp(n);
    Pair<T> * tempPtr = &temp[0];
    for (unsigned i = 0; i < n; ++i)
    {
        tempPtr[i].index = i;
        tempPtr[i].value = values[i];
    }

    // Most of the time is taken with this sort.
    std::sort(temp.begin(), temp.end(), _compare<T>);

    for (unsigned i = 0; i < n; ++i)
    {
        indices[i] = tempPtr[i].index;
        values[i] = tempPtr[i].value;
    }
}

/// Reorder values based on vector of indices.
template <typename T>
void oskar_LinkedSort::reorder(const unsigned n, T * values, const unsigned * indices)
{
    std::vector<T> temp(n);
    for (unsigned i = 0; i < n; ++i)
    {
        temp[i] = values[indices[i]];
    }
    memcpy((void*)values, (const void*)&temp[0], n * sizeof(T));
}

/// Comparison function for sorting indices.
template <typename T>
bool oskar_LinkedSort::_compare(const Pair<T>& l, const Pair<T>& r)
{
    return l.value < r.value;
}

#endif // OSKAR_MATH_LINKED_SORT_H_
