#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <cmath>

/**
 * @brief
 *
 * @details
 */
class Geometry
{
public:
    /// Return the angular distance between two points.
    template<typename T>
    static inline T angularDistance(const T l1, const T l2, const T b1, const T b2);

    /// Convert local horizon coordinates to Cartesian coordinates.
    template<typename T>
    static inline void cartesianFromHorizontal(const T az, const T el,
            T& x, T& y, T& z);

    /// Convert Cartesian coordinates to local horizon coordinates.
    template<typename T>
    static inline void cartesianToHorizontal(const T x, const T y, const T z,
            T& az, T& el);

    /// Computes the polarisation vectors in Cartesian coordinates.
    template<typename T>
    static inline void polarisationEquatorialCartesian(const T polAlpha,
            const T polDelta, const T alpha, const T delta,
            const T coords[6]);

    /// Rotates a point by an angle around an arbitrary axis.
    template<typename T>
    static inline void rotate(const T cosA, const T sinA,
            const T ax, const T ay, const T az,
            const T x0, const T y0, const T z0, T& x, T& y, T& z);

    /// Rotates a point by an angle around the x-axis.
    template<typename T>
    static inline void rotateX(const T cosA, const T sinA,
            const T x0, const T y0, const T z0, T& x, T& y, T& z);

    /// Rotates a point by an angle around the y-axis.
    template<typename T>
    static inline void rotateY(const T cosA, const T sinA,
            const T x0, const T y0, const T z0, T& x, T& y, T& z);

    /// Rotates a point by an angle around the z-axis.
    template<typename T>
    static inline void rotateZ(const T cosA, const T sinA,
            const T x0, const T y0, const T z0, T& x, T& y, T& z);

    /// Computes the 3x3 rotation matrix for a given axis and angle.
    template<typename T>
    static inline void rotationMatrix3(const T cosA, const T sinA,
            const T ax, const T ay, const T az, T m[9]);

    /// Computes the 4x4 rotation matrix for a given axis and angle.
    template<typename T>
    static inline void rotationMatrix4(const T cosA, const T sinA,
            const T ax, const T ay, const T az, T m[16]);

    /// Returns the sign of a number.
    template<typename T>
    static inline T sgn(T x) {return (x < 0) ? -1 : (x > 0) ? 1 : 0;}

    /// Spherical to tangent plane using Orthographic projection.
    template<typename T>
    static inline bool tangentPlaneFromSphericalOrthographic (
            const T longitude, const T latitude, T& x, T& y,
            const T longitude0, const T sinLat0, const T cosLat0);

    /// Tangent plane to spherical using Azimuthal Equidistant projection.
    template<typename T>
    static inline bool tangentPlaneToSphericalAzimuthalEquidistant (
            const T x, const T y, T& longitude, T& latitude,
            const T longitude0, const T sinLat0, const T cosLat0);

    /// Tangent plane to spherical using Gnomonic projection.
    template<typename T>
    static inline bool tangentPlaneToSphericalGnomonic (
            const T x, const T y, T& longitude, T& latitude,
            const T longitude0, const T sinLat0, const T cosLat0);

    /// Tangent plane to spherical using Orthographic projection.
    template<typename T>
    static inline bool tangentPlaneToSphericalOrthographic (
            const T x, const T y, T& longitude, T& latitude,
            const T longitude0, const T sinLat0, const T cosLat0);
};

/**
 * @details
 * Return the angular distance between two points on a sphere.
 * This is implemented using the Haversin formula:
 *
 * \f{equation}{
 *     2\arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right)
 *     +\cos{\phi_s}\cos{\phi_f}\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)
 * \f}
 *
 * @param[in] l1 Longitude of the first point, in radians.
 * @param[in] l2 Longitude of the second point, in radians.
 * @param[in] b1 Latitude of the first point, in radians.
 * @param[in] b2 Latitude of the second point, in radians.
 *
 * @return The angular distance in radians.
 */
template<typename T>
T Geometry::angularDistance(const T l1, const T l2, const T b1, const T b2)
{
    const T delta_l = (l2 - l1) / 2.0;
    const T delta_b = (b2 - b1) / 2.0;
    const T sinDb = sin(delta_b);
    const T sinDl = sin(delta_l);
    return 2 * asin( sqrt(sinDb * sinDb + cos(b1) * cos(b2) * sinDl*sinDl) );
}

/**
* @details Convert local horizon coordinates to Cartesian coordinates.
*
* @param[in] az The azimuth, in radians.
* @param[in] el The elevation, in radians.
* @param[out] x The x-coordinate.
* @param[out] y The y-coordinate.
* @param[out] z The z-coordinate.
*/
template<typename T>
void Geometry::cartesianFromHorizontal(const T az, const T el,
        T& x, T& y, T& z)
{
    const T cosEl = cos(el);
    x = cosEl * sin(az);
    y = cosEl * cos(az);
    z = sin(el);
}

/**
 * @details Convert Cartesian coordinates to local horizon coordinates.
 *
 * @param[in] x The x-coordinate.
 * @param[in] y The y-coordinate.
 * @param[in] z The z-coordinate.
 * @param[out] az The azimuth, in radians.
 * @param[out] el The elevation, in radians.
 */
template<typename T>
void Geometry::cartesianToHorizontal(const T x, const T y, const T z,
        T& az, T& el)
{
    az = atan2(x, y);
    el = atan2(z, sqrt(x * x + y * y) );
}

/**
 * @details
 * Rotate a point by an angle around an arbitrary axis, using the matrix:
 *
 * \f{equation}{
 *       M = \left[\begin{array}{ccc}
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
 * @param[in] cosA Cosine angle of rotation.
 * @param[in] sinA Sine angle of rotation.
 * @param[in] ax   Axis x-component.
 * @param[in] ay   Axis y-component.
 * @param[in] az   Axis z-component.
 * @param[in] x0   Original x-coordinate.
 * @param[in] y0   Original y-coordinate.
 * @param[in] z0   Original z-coordinate.
 * @param[out] x   New x-coordinate.
 * @param[out] y   New y-coordinate.
 * @param[out] z   New z-coordinate.
 */
template<typename T>
void Geometry::rotate(const T cosA, const T sinA,
        const T ax, const T ay, const T az,
        const T x0, const T y0, const T z0, T& x, T& y, T& z)
{
    const T d = 1 - cosA;
    const T xx = d * ax * ax + cosA;
    const T xy = d * ax * ay;
    const T xz = d * ax * az;
    const T yy = d * ay * ay + cosA;
    const T yz = d * ay * az;
    const T zz = d * az * az + cosA;
    const T xs = ax * sinA;
    const T ys = ay * sinA;
    const T zs = az * sinA;

    x = x0 * xx          + y0 * (xy - zs)   + z0 * (xz + ys);
    y = x0 * (xy + zs)   + y0 * yy          + z0 * (yz - xs);
    z = x0 * (xz - ys)   + y0 * (yz + xs)   + z0 * zz;
}

/**
 * @details
 * Rotate a point by an angle around the x-axis.
 *
 * @param[in] cosA Cosine angle of rotation.
 * @param[in] sinA Sine angle of rotation.
 * @param[in] x0   Original x-coordinate.
 * @param[in] y0   Original y-coordinate.
 * @param[in] z0   Original z-coordinate.
 * @param[out] x   New x-coordinate.
 * @param[out] y   New y-coordinate.
 * @param[out] z   New z-coordinate.
 */
template<typename T>
void Geometry::rotateX(const T cosA, const T sinA,
        const T x0, const T y0, const T z0, T& x, T& y, T& z)
{
    x = x0;
    y = y0 * cosA - z0 * sinA;
    z = y0 * sinA + z0 * cosA;
}

/**
 * @details
 * Rotate a point by an angle around the y-axis.
 *
 * @param[in] cosA Cosine angle of rotation.
 * @param[in] sinA Sine angle of rotation.
 * @param[in] x0   Original x-coordinate.
 * @param[in] y0   Original y-coordinate.
 * @param[in] z0   Original z-coordinate.
 * @param[out] x   New x-coordinate.
 * @param[out] y   New y-coordinate.
 * @param[out] z   New z-coordinate.
 */
template<typename T>
void Geometry::rotateY(const T cosA, const T sinA,
        const T x0, const T y0, const T z0, T& x, T& y, T& z)
{
    x = x0 * cosA + z0 * sinA;
    y = y0;
    z = z0 * cosA - x0 * sinA;
}

/**
 * @details
 * Rotate a point by an angle around the z-axis.
 *
 * @param[in] cosA Cosine angle of rotation.
 * @param[in] sinA Sine angle of rotation.
 * @param[in] x0   Original x-coordinate.
 * @param[in] y0   Original y-coordinate.
 * @param[in] z0   Original z-coordinate.
 * @param[out] x   New x-coordinate.
 * @param[out] y   New y-coordinate.
 * @param[out] z   New z-coordinate.
 */
template<typename T>
void Geometry::rotateZ(const T cosA, const T sinA,
        const T x0, const T y0, const T z0, T& x, T& y, T& z)
{
    x = x0 * cosA - y0 * sinA;
    y = x0 * sinA + y0 * cosA;
    z = z0;
}

/**
 * @details
 * Convert coordinates from the spherical system to the tangent plane
 * using the Orthographic projection.
 *
 * The coordinates are:
 *
 * \f{eqnarray}{
 *      x &=& \cos\phi \sin(\lambda - \lambda_0)\\
 *      y &=& \cos\phi_0 \sin\phi -
 *             \sin\phi_0 \cos\phi \cos\left(\lambda - \lambda_0\right)
 * \f}
 *
 * This is equivalent to the SIN projection defined in the FITS standard.
 *
 * @param[in] longitude  The longitude \f$\lambda\f$ on the sphere in radians.
 * @param[in] latitude   The latitude \f$\phi\f$ on the sphere in radians.
 * @param[out] x         The x-coordinate on the tangent plane.
 * @param[out] y         The y-coordinate on the tangent plane.
 * @param[in] longitude0 The longitude \f$\lambda_0\f$ of the tangent point.
 * @param[in] sinLat0    The sine of the latitude \f$\phi_0\f$ of the tangent point.
 * @param[in] cosLat0    The cosine of the latitude \f$\phi_0\f$ of the tangent point.
 *
 * @return The function returns true if the conversion was successful,
 * false if not.
 */
template<typename T>
bool Geometry::tangentPlaneFromSphericalOrthographic (
        const T longitude, const T latitude, T& x, T& y,
        const T longitude0, const T sinLat0, const T cosLat0
){
    const T sinLat = sin(latitude);
    const T cosLat = cos(latitude);
    x = cosLat * sin(longitude - longitude0);
    y = cosLat0 * sinLat - sinLat0 * cosLat * cos(longitude - longitude0);
    return true;
}

/**
 * @details
 * Convert coordinates \e x and \e y from the tangent plane to the
 * spherical system using the Azimuthal Equidistant projection.
 *
 * The longitude \f$\lambda\f$ is given by
 *
 * \f{equation}{
 *      \lambda = \lambda_0 + \arctan\left(\frac{x \sin r}
 *                       {r \cos\phi_0 \cos r - y \sin\phi_0 \sin r}\right)
 * \f}
 *
 * and the latitude \f$\phi\f$ is given by
 *
 * \f{equation}{
 *      \phi = \arcsin\left(y \frac{\cos\phi_0}{f} + \sin\phi_0 \cos r\right)
 * \f}
 *
 * where \f$r = \sqrt{x^2 + y^2}\f$ and \f$f = \frac{r}{\sin r}\f$.
 *
 * This is equivalent to the ARC projection defined in the FITS standard.
 *
 * @param[in] x          The x-coordinate on the tangent plane.
 * @param[in] y          The y-coordinate on the tangent plane.
 * @param[out] longitude The longitude \f$\lambda\f$ on the sphere in radians.
 * @param[out] latitude  The latitude \f$\phi\f$ on the sphere in radians.
 * @param[in] longitude0 The longitude \f$\lambda_0\f$ of the tangent point.
 * @param[in] sinLat0    The sine of the latitude \f$\phi_0\f$ of the tangent point.
 * @param[in] cosLat0    The cosine of the latitude \f$\phi_0\f$ of the tangent point.
 *
 * @return The function returns true if the conversion was successful,
 * false if not.
 */
template<typename T>
bool Geometry::tangentPlaneToSphericalAzimuthalEquidistant (
        const T x, const T y, T& longitude, T& latitude,
        const T longitude0, const T sinLat0, const T cosLat0
){
    const T r = sqrt(x*x + y*y);
    const T sinR = sin(r);
    const T cosR = cos(r);
    const T tiny = 0.001;
    T f = (fabs(r) < tiny) ? sgn(r) : (r / sinR);
    if (fabs(x) < tiny && fabs(y) < tiny) {
        latitude = atan2(sinLat0, cosLat0);
        longitude = longitude0;
    } else {
        latitude = asin(y * cosLat0 / f + sinLat0 * cosR);
        longitude = longitude0 + atan2(x * sinR,
                r * cosLat0 * cosR - y * sinLat0 * sinR);
    }
    return true;
}

/**
 * @details
 * Convert coordinates \e x and \e y from the tangent plane to the
 * spherical system using the Gnomonic projection.
 *
 * The longitude \f$\lambda\f$ is given by
 *
 * \f{equation}{
 *      \lambda = \lambda_0 + \arctan\left(\frac{x}
 *                       {\cos\phi_0 - y \sin\phi_0}\right)
 * \f}
 *
 * and the latitude \f$\phi\f$ is given by
 *
 * \f{equation}{
 *      \phi = \arcsin\left(\frac{\sin\phi_0 + y \cos\phi_0}{s}\right)
 * \f}
 *
 * where \f$s = \sqrt{1 - x^2 + y^2}\f$.
 *
 * This is equivalent to the TAN projection defined in the FITS standard.
 *
 * @param[in] x          The x-coordinate on the tangent plane.
 * @param[in] y          The y-coordinate on the tangent plane.
 * @param[out] longitude The longitude \f$\lambda\f$ on the sphere in radians.
 * @param[out] latitude  The latitude \f$\phi\f$ on the sphere in radians.
 * @param[in] longitude0 The longitude \f$\lambda_0\f$ of the tangent point.
 * @param[in] sinLat0    The sine of the latitude \f$\phi_0\f$ of the tangent point.
 * @param[in] cosLat0    The cosine of the latitude \f$\phi_0\f$ of the tangent point.
 *
 * @return The function returns true if the conversion was successful,
 * false if not.
 */
template<typename T>
bool Geometry::tangentPlaneToSphericalGnomonic (
        const T x, const T y, T& longitude, T& latitude,
        const T longitude0, const T sinLat0, const T cosLat0
){
    const T r2 = x*x + y*y;
    const T s = sqrt(1.0 + r2);
    longitude = longitude0 + atan2(x, cosLat0 - y * sinLat0);
    latitude = asin((sinLat0 + y * cosLat0) / s);
    return true;
}

/**
 * @details
 * Convert coordinates \e x and \e y from the tangent plane to the
 * spherical system using the Orthographic projection.
 *
 * The longitude \f$\lambda\f$ is given by
 *
 * \f{equation}{
 *      \lambda = \lambda_0 + \arctan\left(\frac{x}
 *                       {s \cos\phi_0 - y \sin\phi_0}\right)
 * \f}
 *
 * and the latitude \f$\phi\f$ is given by
 *
 * \f{equation}{
 *      \phi = \arcsin\left(s \sin\phi_0 + y \cos\phi_0\right)
 * \f}
 *
 * where \f$s = \sqrt{1 - x^2 + y^2}\f$.
 *
 * This is equivalent to the SIN projection defined in the FITS standard.
 *
 * @param[in] x          The x-coordinate on the tangent plane.
 * @param[in] y          The y-coordinate on the tangent plane.
 * @param[out] longitude The longitude \f$\lambda\f$ on the sphere in radians.
 * @param[out] latitude  The latitude \f$\phi\f$ on the sphere in radians.
 * @param[in] longitude0 The longitude \f$\lambda_0\f$ of the tangent point.
 * @param[in] sinLat0    The sine of the latitude \f$\phi_0\f$ of the tangent point.
 * @param[in] cosLat0    The cosine of the latitude \f$\phi_0\f$ of the tangent point.
 *
 * @return The function returns true if the conversion was successful,
 * false if not.
 */
template<typename T>
bool Geometry::tangentPlaneToSphericalOrthographic (
        const T x, const T y, T& longitude, T& latitude,
        const T longitude0, const T sinLat0, const T cosLat0
){
    const T r2 = x*x + y*y;
    if (r2 > 1.0) return false;
    const T s = sqrt(1.0 - r2);
    longitude = longitude0 + atan2(x, cosLat0 * s - y * sinLat0);
    latitude = asin(s * sinLat0 + y * cosLat0);
    return true;
}

#endif /* GEOMETRY_H_ */
