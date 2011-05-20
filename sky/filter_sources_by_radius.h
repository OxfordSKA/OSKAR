#ifndef FILTER_SOURCES_BY_RADIUS_H_
#define FILTER_SOURCES_BY_RADIUS_H_

// number of sources is modified by this function.
unsigned filter_sources_by_radius(unsigned * num_sources,
        const double inner_radius, const double outer_radius,
        const double ra0, const double dec0, double * ra, double *dec,
        double * brightness);

// ============================================================================
// private functions...
// ============================================================================
void source_distance_from_phase_centre(const unsigned num_sources,
        const double * ra, const double * dec, const double ra0,
        const double dec0, double * distance);


void reorder_sources(const unsigned num_sources, const unsigned * indices,
        double * ra, double * dec, double * brightness);

#endif // FILTER_SOURCES_BY_RADIUS_H_
