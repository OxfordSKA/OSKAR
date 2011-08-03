#include "sky/filter_sources_by_radius.h"

#include "sky/oskar_LinkedSort.h"

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace std;

unsigned filter_sources_by_radius(unsigned * num_sources,
        const double inner_radius, const double outer_radius,
        const double ra0, const double dec0, double * ra,
        double * dec, double * brightness)
{
    // Convert to radians.
    double inner = inner_radius * (M_PI / 180.0);
    double outer = outer_radius * (M_PI / 180.0);

    // Evaluate the radial distance of sources from the phase centre.
    std::vector<double> dist(*num_sources);
    source_distance_from_phase_centre(*num_sources, ra, dec, ra0, dec0, &dist[0]);

    // Sort the radial distances into increasing order, holding onto
    // the positions of the original indices.
    std::vector<unsigned> indices(*num_sources);
    unsigned * idx = &indices[0];
    oskar_LinkedSort::sortIndices(*num_sources, &dist[0], idx);

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
    reorder_sources(num_remaining_sources, &idx[iStart], ra, dec, brightness);

    // Resize the RA, Dec. and brightness arrays to the number of
    // sources retained by the filter.
    dec = (double*) realloc(dec, num_remaining_sources * sizeof(double));
    ra = (double*) realloc(ra, num_remaining_sources * sizeof(double));
    brightness = (double*) realloc(brightness, num_remaining_sources * sizeof(double));

    // Update the number of sources.
    *num_sources = num_remaining_sources;

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


void reorder_sources(const unsigned num_sources, const unsigned * indices,
        double * ra, double * dec, double * brightness)
{
    std::vector<double> temp(num_sources);
    double * tempPtr = &temp[0];

    // RA
    for (unsigned i = 0; i < num_sources; ++i)
        tempPtr[i] = ra[indices[i]];
    memcpy(ra, tempPtr, num_sources * sizeof(double));

    // Dec.
    for (unsigned i = 0; i < num_sources; ++i)
        tempPtr[i] = dec[indices[i]];
    memcpy(dec, tempPtr, num_sources * sizeof(double));

    // Brightness
    for (unsigned i = 0; i < num_sources; ++i)
        tempPtr[i] = brightness[indices[i]];
    memcpy(brightness, tempPtr, num_sources * sizeof(double));
}
