#ifndef GENERATE_RANDOM_SOURCES_H_
#define GENERATE_RANDOM_SOURCES_H_


void generate_random_sources(const unsigned num_sources, const double inner_radius,
        const double outer_radius, const double brightness_min,
        const double brightness_max, const double distribution_power,
        double * RA, double * Dec, double * brightness, const unsigned seed = 0);


#endif // GENERATE_RANDOM_SOURCES_H_
