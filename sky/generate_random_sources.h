#ifndef GENERATE_RANDOM_SOURCES_H_
#define GENERATE_RANDOM_SOURCES_H_


void generate_random_sources(const unsigned num_sources,
        const double brightness_min, const double brightness_max,
        const double distribution_power, double * ra, double * dec,
        double * brightness, const unsigned seed = 0);


// number of sources is modified by this function.
unsigned filter_sources_by_radius(unsigned * num_sources,
        const double inner_radius, const double outer_radius,
        const double ra0, const double dec0, double * ra, double *dec,
        double * brightness);


// ============================================================================
// private functions...
// ============================================================================
void source_distance(const unsigned num_sources, const double * ra,
        const double * dec, const double ra0, const double dec0,
        double * distance);


void reorder_sources(const unsigned num_sources, const unsigned * indices,
        double * ra, double * dec, double * brightness);




#endif // GENERATE_RANDOM_SOURCES_H_
