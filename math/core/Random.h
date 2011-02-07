#ifndef OSKAR_RANDOM_H_
#define OSKAR_RANDOM_H_

/**
 * @file Random.h
 *
 * @brief This file defines functions to generate random numbers.
 */

#include <cmath>

/**
 * @brief
 * Class containing functions for generating random numbers.
 *
 * @details
 * This class contains functions to generate random numbers from uniform and
 * Gaussian distributions.
 */
class Random
{
public:
    /// Generates a random number from a uniform distribution.
    template<typename T>
    static T uniform(int seed);

    /// Generates a random number from a Gaussian distribution.
    template<typename T>
    static T gaussian(int seed, T* out1, T* out2);
};

/*=============================================================================
 * Static public members
 *---------------------------------------------------------------------------*/

/**
 * @details
 * Generates a random number from a uniform distribution in the range 0 to 1.
 * If the supplied seed is greater than 0, then the random number generator is
 * seeded first.
 *
 * @param[in] seed If greater than 0, use this as a seed for the generator.
 */
template<typename T>
T Random::uniform(int seed)
{
    // Seed the random number generator if required.
    if (seed > 0) srand(seed);
    return (T)rand() / (RAND_MAX + 1.0);
}

/**
 * @details
 * Generates a random number from a Gaussian distribution with zero mean
 * and unit variance.
 *
 * @param[in] seed If greater than 0, use this as a seed for the generator.
 * @param[out] out1 The first random number.
 * @param[out] out2 The second random number.
 */
template<typename T>
T Random::gaussian(int seed, T* x, T* y)
{
    if (seed > 0) srand(seed);
    T r2;
    do {
        // Choose x and y in a uniform square (-1, -1) to (+1, +1).
        *x = 2.0 * rand() / (RAND_MAX + 1.0) - 1.0;
        *y = 2.0 * rand() / (RAND_MAX + 1.0) - 1.0;

        // Check if this is in the unit circle.
        r2 = (*x)*(*x) + (*y)*(*y);
    } while (r2 >= 1.0 || r2 == 0);

    // Box-Muller transform.
    T fac = std::sqrt(-2.0 * std::log(r2) / r2);
    *x *= fac;
    *y *= fac;

    // Return the first random number.
    return *x;
}

#endif // OSKAR_RANDOM_H_
