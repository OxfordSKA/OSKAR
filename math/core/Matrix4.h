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

#ifndef OSKAR_MATRIX4_H_
#define OSKAR_MATRIX4_H_

/**
 * @file Matrix4.h
 *
 * @brief This file defines functions to work with 4x4 matrices.
 */

#include <cmath>

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

/**
 * @brief Functions to work with 4x4 matrices.
 *
 * @details
 * The matrices are stored in column-major order to be compatible with OpenGL
 * matrices and FORTRAN libraries.
 */
class Matrix4
{
public:
    /// Sets up a 4x4 matrix from a 3x3 matrix.
    template<typename T> __device__ __host__
    static inline void fromMatrix3(T matrix[16], const T in[9]);

    /// Returns the identity matrix.
    template<typename T> __device__ __host__
    static inline void identity(T matrix[16]);

    /// Multiplies two 4x4 matrices together.
    template<typename T> __device__ __host__
    static inline void multiplyMatrix4(T result[16],
            const T a[16], const T b[16]);

    /// Multiplies a 3-vector by a 4x4 matrix.
    template<typename T> __device__ __host__
    static inline void multiplyVector3(T result[3],
            const T m[16], const T v[3]);

    /// Multiplies a 4-vector by a 4x4 matrix.
    template<typename T> __device__ __host__
    static inline void multiplyVector4(T result[4],
            const T m[16], const T v[4]);

    /// Returns a matrix for a rotation about a normalised arbitrary axis.
    template<typename T> __device__ __host__
    static inline void rotation(T matrix[16], const T axis[3],
            const T cosA, const T sinA);

    /// Returns a matrix for a rotation around the x-axis.
    template<typename T> __device__ __host__
    static inline void rotationX(T matrix[16], const T cosA, const T sinA);

    /// Returns a matrix for a rotation around the y-axis.
    template<typename T> __device__ __host__
    static inline void rotationY(T matrix[16], const T cosA, const T sinA);

    /// Returns a matrix for a rotation around the z-axis.
    template<typename T> __device__ __host__
    static inline void rotationZ(T matrix[16], const T cosA, const T sinA);

    /// Returns a matrix for a rotation about a normalised arbitrary axis.
    template<typename T> __device__ __host__
    static inline void rotation(T matrix[16], const T axis[3], const T angle);

    /// Returns a matrix for a rotation around the x-axis.
    template<typename T> __device__ __host__
    static inline void rotationX(T matrix[16], const T angle);

    /// Returns a matrix for a rotation around the y-axis.
    template<typename T> __device__ __host__
    static inline void rotationY(T matrix[16], const T angle);

    /// Returns a matrix for a rotation around the z-axis.
    template<typename T> __device__ __host__
    static inline void rotationZ(T matrix[16], const T angle);

    /// Returns a matrix for independent scaling in 3 dimensions.
    template<typename T> __device__ __host__
    static inline void scaling(T matrix[16], const T scale[3]);

    /// Returns a matrix for uniform scaling in 3 dimensions.
    template<typename T> __device__ __host__
    static inline void scaling(T matrix[16], const T scale);

    /// Sets the matrix contents to the given parameters.
    template<typename T> __device__ __host__
    static inline void set(T matrix[16],
            const T, const T, const T, const T,
            const T, const T, const T, const T,
            const T, const T, const T, const T,
            const T, const T, const T, const T);

    /// Returns a translation matrix for the given vector.
    template<typename T> __device__ __host__
    static inline void translation(T matrix[16], const T v[3]);
};

/*=============================================================================
 * Static public members
 *---------------------------------------------------------------------------*/

/**
 * @details
 * Sets up a 4x4 matrix from a 3x3 matrix.
 */
template<typename T> __device__ __host__
void Matrix4::fromMatrix3(T matrix[16], const T in[9])
{
    set<T>(matrix,
            in[0], in[1], in[2], 0,
            in[3], in[4], in[5], 0,
            in[6], in[7], in[8], 0,
            0,     0,     0,     1);
}

/**
 * @details
 * Returns the identity matrix.
 */
template<typename T> __device__ __host__
void Matrix4::identity(T matrix[16])
{
    set<T>(matrix,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
}

/**
 * @details
 * Multiplies the two 4x4 matrices together, to give result = a * b.
 */
template<typename T> __device__ __host__
void Matrix4::multiplyMatrix4(T result[16], const T a[16], const T b[16])
{
#define MMUL(R, C) (b[R]*a[C] + b[R+4]*a[C+1] + b[R+8]*a[C+2] + b[R+12]*a[C+3])
    set<T>(result,
            MMUL(0, 0),  MMUL(1, 0),  MMUL(2, 0),  MMUL(3, 0),
            MMUL(0, 4),  MMUL(1, 4),  MMUL(2, 4),  MMUL(3, 4),
            MMUL(0, 8),  MMUL(1, 8),  MMUL(2, 8),  MMUL(3, 8),
            MMUL(0, 12), MMUL(1, 12), MMUL(2, 12), MMUL(3, 12));
#undef MMUL
}

/**
 * @details
 * Multiplies a 3-vector by a 4x4 matrix.
 */
template<typename T> __device__ __host__
void Matrix4::multiplyVector3(T result[3], const T m[16], const T v[3])
{
    result[0] = m[0]*v[0] + m[4]*v[1] + m[8]*v[2]  + m[12];
    result[1] = m[1]*v[0] + m[5]*v[1] + m[9]*v[2]  + m[13];
    result[2] = m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14];
}

/**
 * @details
 * Multiplies a 4-vector by a 4x4 matrix.
 */
template<typename T> __device__ __host__
void Matrix4::multiplyVector4(T result[4], const T m[16], const T v[4])
{
    result[0] = m[0]*v[0] + m[4]*v[1] +  m[8]*v[2] + m[12]*v[3];
    result[1] = m[1]*v[0] + m[5]*v[1] +  m[9]*v[2] + m[13]*v[3];
    result[2] = m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3];
    result[3] = m[3]*v[0] + m[6]*v[1] + m[11]*v[2] + m[15]*v[3];
}

/**
 * @details
 * Compute a 4x4 matrix corresponding to a spatial translation,
 * according to
 *
 * \f{equation}{
 *       m = \left[\begin{array}{cccc}
 *               1 & 0 & 0 & x \\
 *               0 & 1 & 0 & y \\
 *               0 & 0 & 1 & z \\
 *               0 & 0 & 0 & 1 \\
 *           \end{array}\right]
 * \f}
 *
 * where \f$x\f$, \f$y\f$ and \f$z\f$ are the offsets in the \f$x\f$,
 * \f$y\f$ and \f$z\f$ dimensions.
 *
 * @param[in] v The translation 3-vector.
 */
template<typename T> __device__ __host__
void Matrix4::translation(T matrix[16], const T v[3])
{
    set<T>(matrix,
            1,      0,      0,      0,
            0,      1,      0,      0,
            0,      0,      1,      0,
            v[0],   v[1],   v[2],   1);
}

/**
 * @details
 * Compute a 4x4 matrix corresponding to a rotation of an angle with
 * cosine \f$c\f$ and sine \f$s\f$ about a unit-length axis, according to:
 *
 * \f{equation}{
 *       m = \left[\begin{array}{cccc}
 *               xx(1-c)+c  & xy(1-c)-zs & xz(1-c)+ys & 0 \\
 *               yx(1-c)+zs & yy(1-c)+c  & yz(1-c)-xs & 0 \\
 *               xz(1-c)-ys & yz(1-c)+xs & zz(1-c)+c  & 0 \\
 *               0          & 0          & 0          & 1 \\
 *           \end{array}\right]
 * \f}
 *
 * where \f$x\f$ is the \f$x\f$-component of the rotation axis,
 * \f$y\f$ is the \f$y\f$-component of the rotation axis and
 * \f$z\f$ is the \f$z\f$-component of the rotation axis.
 *
 * The axis must be normalised to length 1 prior to calling this function.
 *
 * @param[in]  a  A unit 3-vector describing the axis of rotation.
 * @param[in]  c  Cosine angle of rotation.
 * @param[in]  s  Sine angle of rotation.
 */
template<typename T> __device__ __host__
void Matrix4::rotation(T matrix[16], const T a[3], const T c, const T s)
{
    const T d = 1 - c;
    const T xx = d * a[0] * a[0];
    const T xy = d * a[0] * a[1];
    const T xz = d * a[0] * a[2];
    const T yy = d * a[1] * a[1];
    const T yz = d * a[1] * a[2];
    const T zz = d * a[2] * a[2];
    const T xs = a[0] * s;
    const T ys = a[1] * s;
    const T zs = a[2] * s;

    set<T>(matrix,
            xx + c,  xy + zs, xz - ys, 0,
            xy - zs, yy + c,  yz + xs, 0,
            xz + ys, yz - xs, zz + c,  0,
            0,       0,       0,       1);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle around the \e axis.
 * The axis must be already normalised.
 */
template<typename T> __device__ __host__
void Matrix4::rotation(T matrix[16], const T axis[3], const T angle)
{
    const T c = (T) cos(angle);
    const T s = (T) sin(angle);
    rotation<T>(matrix, axis, c, s);
}

/**
 * @details
 * Returns a matrix for a rotation about the x-axis by an angle with
 * cosine \e c and sine \e s.
 */
template<typename T> __device__ __host__
void Matrix4::rotationX(T matrix[16], const T c, const T s)
{
    set<T>(matrix,
            1, 0, 0, 0,
            0, c, s, 0,
            0,-s, c, 0,
            0, 0, 0, 1);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle about the x-axis.
 */
template<typename T> __device__ __host__
void Matrix4::rotationX(T matrix[16], const T angle)
{
    const T c = (T) cos(angle);
    const T s = (T) sin(angle);
    rotationX<T>(matrix, c, s);
}

/**
 * @details
 * Returns a matrix for a rotation about the y-axis by an angle with
 * cosine \e c and sine \e s.
 */
template<typename T> __device__ __host__
void Matrix4::rotationY(T matrix[16], const T c, const T s)
{
    set<T>(matrix,
            c, 0,-s, 0,
            0, 1, 0, 0,
            s, 0, c, 0,
            0, 0, 0, 1);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle about the y-axis.
 */
template<typename T> __device__ __host__
void Matrix4::rotationY(T matrix[16], const T angle)
{
    const T c = (T) cos(angle);
    const T s = (T) sin(angle);
    rotationY<T>(matrix, c, s);
}

/**
 * @details
 * Returns a matrix for a rotation about the z-axis by an angle with
 * cosine \e c and sine \e s.
 */
template<typename T> __device__ __host__
void Matrix4::rotationZ(T matrix[16], const T c, const T s)
{
    set<T>(matrix,
            c,  s, 0, 0,
            -s, c, 0, 0,
            0,  0, 1, 0,
            0,  0, 0, 1);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle about the z-axis.
 */
template<typename T> __device__ __host__
void Matrix4::rotationZ(T matrix[16], const T angle)
{
    const T c = (T) cos(angle);
    const T s = (T) sin(angle);
    rotationZ<T>(matrix, c, s);
}

/**
 * @details
 * Compute a 4x4 matrix corresponding to a spatial scale, according to:
 *
 * \f{equation}{
 *       m = \left[\begin{array}{cccc}
 *               x & 0 & 0 & 0 \\
 *               0 & y & 0 & 0 \\
 *               0 & 0 & z & 0 \\
 *               0 & 0 & 0 & 1 \\
 *           \end{array}\right]
 * \f}
 *
 * where \f$x\f$, \f$y\f$ and \f$z\f$ are the spatial scales in the \f$x\f$,
 * \f$y\f$ and \f$z\f$ dimensions.
 *
 * @param[in]  s The scale 3-vector.
 */
template<typename T> __device__ __host__
void Matrix4::scaling(T matrix[16], const T s[3])
{
    set<T>(matrix,
            s[0], 0,    0,    0,
            0,    s[1], 0,    0,
            0,    0,    s[2], 0,
            0,    0,    0,    1);
}

/**
 * @details
 * Returns a matrix for uniform scaling by \e scale in 3 dimensions.
 */
template<typename T> __device__ __host__
void Matrix4::scaling(T matrix[16], const T scale)
{
    T vec[3] = {scale, scale, scale};
    scaling<T>(matrix, vec);
}

/**
 * @details
 * Sets the contents of the matrix.
 * Note that the elements are given in column-major (FORTRAN) order.
 */
template<typename T> __device__ __host__
void Matrix4::set(T matrix[16],
        const T a, const T b, const T c, const T d,
        const T e, const T f, const T g, const T h,
        const T i, const T j, const T k, const T l,
        const T m, const T n, const T o, const T p)
{
    matrix[0] = a; matrix[4] = e; matrix[8]  = i; matrix[12] = m;
    matrix[1] = b; matrix[5] = f; matrix[9]  = j; matrix[13] = n;
    matrix[2] = c; matrix[6] = g; matrix[10] = k; matrix[14] = o;
    matrix[3] = d; matrix[7] = h; matrix[11] = l; matrix[15] = p;
}

#endif /* OSKAR_MATRIX4_H_ */
