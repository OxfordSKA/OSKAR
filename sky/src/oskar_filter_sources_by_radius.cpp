#include "sky/oskar_filter_sources_by_radius.h"

#include "math/oskar_LinkedSort.h"

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdio>

using namespace std;

unsigned filter_sources_by_radius(unsigned * num_sources,
        const double inner_radius, const double outer_radius,
        const double ra0, const double dec0, double** ra,
        double** dec, double** brightness)
{
    // Convert to radians.
    double inner = inner_radius * (M_PI / 180.0);
    double outer = outer_radius * (M_PI / 180.0);

    // Evaluate the radial distance of sources from the phase centre.
    double* dist = (double*)malloc(*num_sources * sizeof(double));
    source_distance_from_phase_centre(*num_sources, *ra, *dec, ra0, dec0, dist);

    // Sort the radial distances into increasing order, holding onto
    // the positions of the original indices.
    unsigned* idx = (unsigned*)malloc(*num_sources * sizeof(unsigned));
    oskar_LinkedSort::sortIndices(*num_sources, dist, idx);

    // Find the indices of the sorted radial distance distance array
    // corresponding to the inner and outer radius.
    unsigned iStart = 0;
    for (unsigned i = 0; i < *num_sources; ++i)
    {
        if (dist[i] >= inner)
        {
            iStart = i;
            break;
        }
    }
    unsigned iEnd = 0;
    for (unsigned i = *num_sources-1;(i >= iStart && i < *num_sources); --i)
    {
        if (dist[i] <= outer)
        {
            iEnd = i;
            break;
        }
    }
    const unsigned num_remaining_sources = iEnd - iStart + 1;

    // Re-order the RA, Dec. and Brightness arrays for the remaining sources
    // according to distance.
    reorder_sources(num_remaining_sources, &idx[iStart], *ra, *dec, *brightness);

    // Resize the RA, Dec. and brightness arrays to the number of
    // sources retained by the filter.
    size_t mem_size = num_remaining_sources * sizeof(double);
    *dec = (double*)realloc(*dec, mem_size);
    *ra  = (double*)realloc(*ra, mem_size);
    *brightness = (double*)realloc(*brightness, mem_size);

    // Update the number of sources.
    *num_sources = num_remaining_sources;

    free(dist);
    free(idx);

    return num_remaining_sources;
}


void source_distance_from_phase_centre(const unsigned num_sources,
        const double * ra, const double * dec, const double ra0,
        const double dec0, double * distance)
{
    const double cosDec0 = cos(dec0);
    const double sinDec0 = sin(dec0);
    for (unsigned i = 0; i < num_sources; ++i)
    {
        distance[i] = acos(sinDec0 * sin(dec[i]) + cosDec0 * cos(dec[i]) *
                cos(ra[i] - ra0));
    }
}


void reorder_sources(const unsigned num_sources, const unsigned* indices,
        double* ra, double* dec, double* brightness)
{
    double* temp = (double*)malloc(num_sources * sizeof(double));

    // RA
    for (unsigned i = 0; i < num_sources; ++i)
        temp[i] = ra[indices[i]];
    memcpy(ra, temp, num_sources * sizeof(double));

    // Dec.
    for (unsigned i = 0; i < num_sources; ++i)
        temp[i] = dec[indices[i]];
    memcpy(dec, temp, num_sources * sizeof(double));

    // Brightness
    for (unsigned i = 0; i < num_sources; ++i)
        temp[i] = brightness[indices[i]];
    memcpy(brightness, temp, num_sources * sizeof(double));

    free(temp);
}
