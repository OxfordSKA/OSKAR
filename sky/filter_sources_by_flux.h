#ifndef FILTER_SOURCES_BY_FLUX_H_
#define FILTER_SOURCES_BY_FLUX_H_

// number of sources is modified by this function.
unsigned filter_sources_by_flux(unsigned * num_sources,
        const double min_brightness, const double max_brightness,
        double * ra, double *dec, double * brightness);

#endif // FILTER_SOURCES_BY_FLUX_H_
