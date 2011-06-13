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

#ifndef MATRIX_H_
#define MATRIX_H_

/**
* @file Matrix.h
*/

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace oskar {

/**
 * @class Matrix
 *
 * @brief
 * Container class for a 2-dimensional array sorted in linear memory.
 *
 * @details
 * Matrix container with dimensions nRows (y) x nColumns (x).
 *
 * The matrix is ordered in 'C' memory order with x or columns being the
 * fastest varying dimension.
 *
 * Access is provided by means of:
 * - operators Maxtrix[y][x] and Matrix(y, x)
 * - ptr() which returns a 2-dimensional pointer.
 * - arrayPtr() which returns a 1-dimensional pointer.
 *
 * A number of basic utility methods are provided.
 */

template <typename T> class Matrix
{
    public:
        /// Constructs an empty matrix
        Matrix() : _nCols(0), _nRows(0), _M(0), _a(0) {};

        /// Constructs a matrix of the specified size.
        Matrix(unsigned nRows, unsigned nCols);

        /// Copies the matrix.
        Matrix(Matrix& m);

        /// Destroys the matrix cleaning up memory.
        virtual ~Matrix() { clear(); }

    public:
        /// Resize the matrix to the specified dimensions.
        void resize(unsigned nRows, unsigned nCols);

        /// Resize the matrix and assign all entries to the value specified.
        void resize(unsigned nRows, unsigned nCols, T value);

        /// Fills every entry in the matrix with the value specified.
        void fill(T value);

        /// Returns true if the Matrix is empty.
        bool empty() const
        { return (_nCols == 0 || _nRows == 0) ? true : false; }

        /// Clear the matrix values and data.
        void clear();

        /// Memory size used the the Matrix in bytes.
        size_t mem() const
        { return _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*); }

        /// Scale the grid by the specified value.
        void scale(const T value);

        unsigned size() const { return _nCols * _nRows; }
        unsigned nColumns() const { return _nCols; }
        unsigned nX() const { return _nCols; }
        unsigned nRows() const { return _nRows; }
        unsigned nY() const { return _nRows; }

        Matrix& matrix() { return *this; }
        const Matrix& matrix() const { return *this; }

        T** ptr() { return _M; }
        T const * const * ptr() const { return _M; }

        T const * arrayPtr() const { return _a; }
        T * arrayPtr() { return _a; }

        T const * rowPtr(unsigned row) const { return _M[row]; }
        T* rowPtr(unsigned row) { return _M[row]; }

        T const * operator [] (unsigned row) const { return _M[row]; }
        T * operator [] (unsigned row) { return _M[row]; }

        T operator () (unsigned row, unsigned col) const { return _M[row][col]; }
        T& operator () (unsigned row, unsigned col) { return _M[row][col]; }

        /// Assignment operator (M1 = M2).
        Matrix& operator = (const Matrix& other);

        /// Flip left-right.
        void fliplr();

        /// Flip up-down.
        void flipud();

        /// Return the minimum of the matrix.
        T min() const;

        /// Return the maximum of the matrix.
        T max() const;

        /// Return the sum of all element of the the matrix.
        T sum() const;

    private:
        unsigned _nCols;  // Fastest varying dimension (x)
        unsigned _nRows;  // Slower varying dimension (y)
        T** _M; // Matrix pointer _M[y][x].
        T* _a;  // Array pointer. _a[y * _nCols + x]
};



//
//------------------------------------------------------------------------------
// Inline method/function definitions.
//

template <typename T>
inline
Matrix<T>::Matrix(unsigned nRows, unsigned nCols)
: _nCols(nCols), _nRows(nRows), _M(0), _a(0)
{
    size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
    _M = (T**) malloc(size);
    unsigned dp = _nRows * (unsigned)sizeof(T*) / (unsigned)sizeof(T);
    for (unsigned y = 0; y < _nRows; y++)
    {
        _M[y] = (T*)_M + dp + y * _nCols;
    }
    _a = (T*)_M + dp;
}

template <typename T>
inline
Matrix<T>::Matrix(Matrix& m)
{
    _nRows = m._nRows;
    _nCols = m._nCols;
    size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
    _M = (T**) malloc(size);
    memcpy((void*)_M, (void*)m._M, size);
    // Re-construct the lookup table pointers (so they don't point to the old data!)
    unsigned dp = _nRows * (unsigned)sizeof(T*) / (unsigned)sizeof(T);
    for (unsigned y = 0; y < _nRows; ++y)
    {
        _M[y] = (T*)_M + dp + y * _nCols;
    }
    _a = (T*)_M + dp;
}


template <typename T>
inline
void Matrix<T>::resize(unsigned nRows, unsigned nCols)
{
    // check if we need to resize
    if (nRows != 0 && nRows == _nRows && nCols != 0 && nCols == _nCols)
        return;
    _nRows = nRows;
    _nCols = nCols;
    size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
    _M = (T**) realloc(_M, size);
    unsigned dp = _nRows * (unsigned)sizeof(T*) / (unsigned)sizeof(T);
    for (unsigned y = 0; y < _nRows; ++y)
    {
        _M[y] = (T*)_M + dp + y * _nCols;
    }
    _a = (T*)_M + dp;
}


template <typename T>
inline
void Matrix<T>::resize(unsigned nRows, unsigned nCols, T value)
{
    resize(nRows, nCols);
    fill(value);
}

template <typename T>
inline
void Matrix<T>::fill(T value)
{
    for (unsigned i = 0; i < _nRows * _nCols; ++i)
        _a[i] = value;
}


template <typename T>
inline
void Matrix<T>::clear()
{
    _nRows = _nCols = 0;
    if (_M) { free(_M); _M = 0; }
    _a = 0;
}


template <typename T>
inline
void Matrix<T>::scale(const T value)
{
    for (unsigned i = 0; i < _nRows * _nCols; ++i)
        _a[i] *= value;
}


template <typename T>
inline
Matrix<T>& Matrix<T>::operator = (const Matrix<T>& other)
{
    if (this != &other)
    {
        clear(); // this can be faster (don't need to clear always see vector header)
        _nCols = other._nCols; _nRows = other._nRows;
        size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
        _M = (T**) malloc(size);
        memcpy((void*)_M, (void*)other._M, size);
        // Re-construct the lookup table pointers (so they don't point to the old data!)
        unsigned dp = _nRows * sizeof(T*) / sizeof(T);
        for (unsigned y = 0; y < _nRows; ++y) {
            _M[y] = (T*)_M + dp + y * _nCols;
        }
        _a = (T*)_M + dp;
    }
    return *this;
}


template <typename T>
inline
void Matrix<T>::fliplr()
{
    size_t rowSize = _nCols * sizeof(T);
    T* tempRow = (T*) malloc(rowSize);
    for (unsigned y = 0; y < _nRows; ++y) {
        for (unsigned x = 0; x < _nCols; ++x) {
            tempRow[x] = _M[y][_nCols - x - 1];
        }
        memcpy(_M[y], tempRow, rowSize);
    }
    free (tempRow);
}


template <typename T>
inline
void Matrix<T>::flipud()
{
    size_t rowSize = _nCols * sizeof(T);
    T* tempRow = (T*) malloc(rowSize);
    unsigned yDest = 0;
    for (unsigned y = 0; y < floor(_nRows / 2); ++y) {
        yDest = _nRows - y - 1;
        memcpy(tempRow, _M[y], rowSize);
        memcpy(_M[y], _M[yDest], rowSize);
        memcpy(_M[yDest], tempRow, rowSize);
    }
    free (tempRow);
}

template <typename T>
inline
T Matrix<T>::min() const
{
    T min = _a[0];
    for (unsigned i = 0; i < _nRows * _nCols; ++i) min = std::min<T>(_a[i], min);
    return min;
}

template <typename T>
inline
T Matrix<T>::max() const
{
    T max = _a[0];
    for (unsigned i = 0; i < _nRows * _nCols; ++i) max = std::max<T>(_a[i], max);
    return max;
}

template <typename T>
inline
T Matrix<T>::sum() const
{
    T sum = 0.0;
    for (unsigned i = 0; i < _nRows * _nCols; ++i) sum += _a[i];
    return sum;
}


} // namespace oskar
#endif // MATRIX_H_
