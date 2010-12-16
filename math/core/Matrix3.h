#ifndef OSKAR_MATRIX3_H_
#define OSKAR_MATRIX3_H_

/**
 * @file Matrix3.h
 *
 * @brief This file defines functions to work with 3x3 matrices.
 */

#include <cmath>

/**
 * @brief Functions to work with 3x3 matrices.
 *
 * @details
 * The matrices are stored in column-major order to be compatible with OpenGL
 * matrices and FORTRAN libraries.
 */
class Matrix3
{
public:
    /// Returns the identity matrix.
    template<typename T>
    static inline void identity(T matrix[9]);

    /// Multiplies two 3x3 matrices together.
    template<typename T>
    static inline void multiplyMatrix3(T result[9],
            const T a[9], const T b[9]);

    /// Multiplies a 3-vector by a 3x3 matrix.
    template<typename T>
    static inline void multiplyVector3(T result[3],
            const T m[9], const T v[3]);

    /// Returns a matrix for a rotation about a normalised arbitrary axis.
    template<typename T>
    static inline void rotation(T matrix[9], const T axis[3],
            const T cosA, const T sinA);

    /// Returns a matrix for a rotation around the x-axis.
    template<typename T>
    static inline void rotationX(T matrix[9], const T cosA, const T sinA);

    /// Returns a matrix for a rotation around the y-axis.
    template<typename T>
    static inline void rotationY(T matrix[9], const T cosA, const T sinA);

    /// Returns a matrix for a rotation around the z-axis.
    template<typename T>
    static inline void rotationZ(T matrix[9], const T cosA, const T sinA);

    /// Returns a matrix for a rotation about a normalised arbitrary axis.
    template<typename T>
    static inline void rotation(T matrix[9], const T axis[3], const T angle);

    /// Returns a matrix for a rotation around the x-axis.
    template<typename T>
    static inline void rotationX(T matrix[9], const T angle);

    /// Returns a matrix for a rotation around the y-axis.
    template<typename T>
    static inline void rotationY(T matrix[9], const T angle);

    /// Returns a matrix for a rotation around the z-axis.
    template<typename T>
    static inline void rotationZ(T matrix[9], const T angle);

    /// Returns a matrix for independent scaling in 3 dimensions.
    template<typename T>
    static inline void scaling(T matrix[9], const T scale[3]);

    /// Returns a matrix for uniform scaling in 3 dimensions.
    template<typename T>
    static inline void scaling(T matrix[9], const T scale);

    /// Sets the matrix contents to the given parameters.
    template<typename T>
    static inline void set(T matrix[9],
            const T, const T, const T,
            const T, const T, const T,
            const T, const T, const T);
};

/*=============================================================================
 * Static public members
 *---------------------------------------------------------------------------*/

/**
 * @details
 * Returns the identity matrix.
 */
template<typename T>
void Matrix3::identity(T matrix[9])
{
    set<T>(matrix,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1);
}

/**
 * @details
 * Multiplies the two 3x3 matrices together, to give result = a * b.
 */
template<typename T>
void Matrix3::multiplyMatrix3(T result[9], const T a[9], const T b[9])
{
#define MMUL(R, C) (b[R]*a[C] + b[R+3]*a[C+1] + b[R+6]*a[C+2])
    set<T>(result,
            MMUL(0, 0),  MMUL(1, 0),  MMUL(2, 0),
            MMUL(0, 3),  MMUL(1, 3),  MMUL(2, 3),
            MMUL(0, 6),  MMUL(1, 6),  MMUL(2, 6));
#undef MMUL
}

/**
 * @details
 * Multiplies a 3-vector by a 3x3 matrix.
 */
template<typename T>
void Matrix3::multiplyVector3(T result[3], const T m[9], const T v[3])
{
    result[0] = m[0]*v[0] + m[3]*v[1] + m[6]*v[2];
    result[1] = m[1]*v[0] + m[4]*v[1] + m[7]*v[2];
    result[2] = m[2]*v[0] + m[5]*v[1] + m[8]*v[2];
}

/**
 * @details
 * Compute a 3x3 matrix corresponding to a rotation of an angle with
 * cosine \f$c\f$ and sine \f$s\f$ about a unit-length axis, according to:
 *
 * \f{equation}{
 *       m = \left[\begin{array}{cccc}
 *               xx(1-c)+c  & xy(1-c)-zs & xz(1-c)+ys \\
 *               yx(1-c)+zs & yy(1-c)+c  & yz(1-c)-xs \\
 *               xz(1-c)-ys & yz(1-c)+xs & zz(1-c)+c  \\
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
template<typename T>
void Matrix3::rotation(T matrix[9], const T a[3], const T c, const T s)
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
            xx + c,  xy + zs, xz - ys,
            xy - zs, yy + c,  yz + xs,
            xz + ys, yz - xs, zz + c);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle around the \e axis.
 * The axis must be already normalised.
 */
template<typename T>
void Matrix3::rotation(T matrix[9], const T axis[3], const T angle)
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
template<typename T>
void Matrix3::rotationX(T matrix[9], const T c, const T s)
{
    set<T>(matrix,
            1, 0, 0,
            0, c, s,
            0,-s, c);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle about the x-axis.
 */
template<typename T>
void Matrix3::rotationX(T matrix[9], const T angle)
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
template<typename T>
void Matrix3::rotationY(T matrix[9], const T c, const T s)
{
    set<T>(matrix,
            c, 0,-s,
            0, 1, 0,
            s, 0, c);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle about the y-axis.
 */
template<typename T>
void Matrix3::rotationY(T matrix[9], const T angle)
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
template<typename T>
void Matrix3::rotationZ(T matrix[9], const T c, const T s)
{
    set<T>(matrix,
            c,  s, 0,
            -s, c, 0,
            0,  0, 1);
}

/**
 * @details
 * Returns a matrix for a rotation by an \e angle about the z-axis.
 */
template<typename T>
void Matrix3::rotationZ(T matrix[9], const T angle)
{
    const T c = (T) cos(angle);
    const T s = (T) sin(angle);
    rotationZ<T>(matrix, c, s);
}

/**
 * @details
 * Compute a 3x3 matrix corresponding to a spatial scale, according to:
 *
 * \f{equation}{
 *       m = \left[\begin{array}{cccc}
 *               x & 0 & 0 \\
 *               0 & y & 0 \\
 *               0 & 0 & z \\
 *           \end{array}\right]
 * \f}
 *
 * where \f$x\f$, \f$y\f$ and \f$z\f$ are the spatial scales in the \f$x\f$,
 * \f$y\f$ and \f$z\f$ dimensions.
 *
 * @param[in]  s The scale 3-vector.
 */
template<typename T>
void Matrix3::scaling(T matrix[9], const T s[3])
{
    set<T>(matrix,
            s[0], 0,    0,
            0,    s[1], 0,
            0,    0,    s[2]);
}

/**
 * @details
 * Returns a matrix for uniform scaling by \e scale in 3 dimensions.
 */
template<typename T>
void Matrix3::scaling(T matrix[9], const T scale)
{
    T vec[3] = {scale, scale, scale};
    scaling<T>(matrix, vec);
}

/**
 * @details
 * Sets the contents of the matrix.
 * Note that the elements are given in column-major (FORTRAN) order.
 */
template<typename T>
void Matrix3::set(T matrix[9],
        const T a, const T b, const T c,
        const T d, const T e, const T f,
        const T g, const T h, const T i)
{
    matrix[0] = a; matrix[3] = d; matrix[6] = g;
    matrix[1] = b; matrix[4] = e; matrix[7] = h;
    matrix[2] = c; matrix[5] = f; matrix[8] = i;
}

#endif /* OSKAR_MATRIX3_H_ */
