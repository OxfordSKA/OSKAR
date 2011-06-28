#include "sky/generate_random_sources.h"

#include <cstdlib>
#include <cmath>

using namespace std;

/**
 * @details
 * Generates a set of sources at random positions on the sky with a brightness
 * distribution according to a power law.
 *
 * Power law random numbers generated according to:
 * http://mathworld.wolfram.com/RandomNumber.html
 *
 * @param[in]  num_sources         The number of sources to generate.
 * @param[in]  brightness_min      The minimum source flux
 * @param[in]  brightness_max
 * @param[in]  distribution_power
 * @param[out] ra
 * @param[out] dec
 * @param[out] brightness
 * @param[in]  seed
 */
void generate_random_sources(const unsigned num_sources,
        const double brightness_min, const double brightness_max,
        const double distribution_power, double * ra, double * dec,
        double * brightness, const unsigned seed)
{
    // seed the random number generator
    srand(seed);

    // power law slope parameter;
    const double p = distribution_power + 1;

    // generate random sources.
    for (unsigned i = 0; i < num_sources; ++i)
    {
        const double r1 = (double)rand() / ((double)RAND_MAX + 1.0);
        const double r2 = (double)rand() / ((double)RAND_MAX + 1.0);
        const double r3 = (double)rand() / ((double)RAND_MAX + 1.0);
        dec[i] = M_PI / 2.0 - acos(2.0 * r1 - 1);
        ra[i] = 2.0 * M_PI * r2;
        const double b0 = pow(brightness_min, p);
        const double b1 = pow(brightness_max, p);
        brightness[i] = pow( ((b1 - b0) * r3 + b0), (1.0 / p));
    }
}

