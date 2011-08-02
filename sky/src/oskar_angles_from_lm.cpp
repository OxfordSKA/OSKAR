#include "sky/oskar_angles_from_lm.h"

#include <limits>
#include <cmath>
using std::numeric_limits;
using std::sqrt;
using std::cos;
using std::sin;
using std::asin;
using std::atan2;
using std::fabs;


/**
 * Returns the right ascension and declination of the supplied array of
 * cosine directions (l and m).
 *
 * @param[in]  num_positions Number of positions to evaluate.
 * @param[in]  ra0           Right Ascension of the field centre, in radians.
 * @param[in]  dec0          Declination of the field centre, in radians.
 * @param[in]  l             Array of positions in cosine space.
 * @param[in]  m             Array of positions in cosine space.
 * @param[out] ra            Array of Right Ascension values, in radians.
 * @param[out] dec           Array of Declination values, in radians.
 */
void oskar_angles_from_lm(const unsigned num_positions, const double ra0,
        const double dec0, const double * l, const double * m, double * ra,
        double * dec)
{
    const double sinDec0 = sin(dec0);
    const double cosDec0 = cos(dec0);

    // Loop over l, m positions and evaluate the Ra, Dec values.
    for (unsigned i = 0; i < num_positions; ++i)
    {
        const double p = sqrt(m[i] * m[i] + l[i] * l[i]);
        const double c = asin(p);

        const double y = l[i] * sin(c);
        const double x = p * cosDec0 * cos(c) + m[i] * sinDec0 * sin(c);
        ra[i] = ra0 + atan2(y, x);

        // Catch divide by zero error at the field centre.
        if (fabs(p) < numeric_limits<double>::epsilon())
            dec[i] = M_PI / 2.0;
        else
            dec[i] = asin((cos(c) * sinDec0) + (m[i] * sin(c) * cosDec0) / p);
    }
}


/**
 * Returns the right ascension and declination of the supplied array of
 * cosine directions (l and m).
 *
 * @param[in]  num_positions Number of positions to evaluate.
 * @param[in]  l             Array of positions in cosine space.
 * @param[in]  m             Array of positions in cosine space.
 * @param[out] ra            Array of Right Ascension values, in radians.
 * @param[out] dec           Array of Declination values, in radians.
 */
void oskar_angles_from_lm_unrotated(const unsigned num_positions, const double * l,
        const double * m, double * ra, double * dec)
{
    // Loop over l, m positions and evaluate the Ra, Dec values.
    for (unsigned i = 0; i < num_positions; ++i)
    {
        const double p = sqrt(m[i] * m[i] + l[i] * l[i]);
        const double c = asin(p);

        const double y = l[i] * sin(c);
        const double x = m[i] * sin(c);
        ra[i] = atan2(y, x);

        // Catch divide by zero error at the field centre.
        if (fabs(p) < numeric_limits<double>::epsilon())
            dec[i] = M_PI / 2.0;
        else
            dec[i] = asin( cos(c) / p);
    }
}

