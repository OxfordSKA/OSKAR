#ifndef GENERATE_RANDOM_SOURCES_H_
#define GENERATE_RANDOM_SOURCES_H_

/**
 * Generates random sources from a power law distribution.
 */
void generate_random_sources(const unsigned num_sources,
        const double brightness_min, const double brightness_max,
        const double distribution_power, double * ra, double * dec,
        double * brightness, const unsigned seed = 0);


#endif // GENERATE_RANDOM_SOURCES_H_
