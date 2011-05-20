#include "sky/rotate_sources.h"

#include <cstring>
#include <cmath>
using std::cos;
using std::sin;
using std::atan2;
using std::sqrt;

void rotate_sources_to_phase_centre(const unsigned num_sources,
        double * ra, double * dec, const double ra0, const double dec0)
{
    // Construct the rotation matrix.
    const double sinRa0 = sin(ra0);
    const double cosRa0 = cos(ra0);
    const double sinDec0 = sin(dec0);
    const double cosDec0 = cos(dec0);

    double rotation_matrix[9];

    rotation_matrix[0] = cosRa0 * sinDec0;
    rotation_matrix[1] = -sinRa0;
    rotation_matrix[2] = cosRa0 * cosDec0;

    rotation_matrix[3] = sinRa0 * sinDec0;
    rotation_matrix[4] = cosRa0;
    rotation_matrix[5] = sinRa0 * cosDec0;

    rotation_matrix[6] = -cosDec0;
    rotation_matrix[7] = 0;
    rotation_matrix[8] = sinDec0;

    // Iterate over and transform the source coordinates.
    double s[3]; // source position (x, y, z)
    for (unsigned i = 0; i < num_sources; ++i)
    {
        s[0] = cos(dec[i]) * cos(ra[i]); // x
        s[1] = cos(dec[i]) * sin(ra[i]); // y
        s[2] = sin(dec[i]);              // z

        mult_matrix_vector(rotation_matrix, s);

        ra[i] = atan2( s[1], s[0] );
        dec[i] = atan2( s[2], sqrt(s[0] * s[0] + s[1] * s[1]) );
    }
}


void mult_matrix_vector(const double * M, double * v)
{
    double s[3];
    s[0] = (M[0] * v[0]) + (M[1] * v[1]) + (M[2] * v[2]);
    s[1] = (M[3] * v[0]) + (M[4] * v[1]) + (M[5] * v[2]);
    s[2] = (M[6] * v[0]) + (M[7] * v[1]) + (M[8] * v[2]);
    memcpy((void*)v, (const void*)s, 3 * sizeof(double));
}
