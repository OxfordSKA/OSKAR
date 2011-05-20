#include "sky/filter_sources_by_flux.h"
#include "sky/filter_sources_by_radius.h"

#include "sky/LinkedSort.h"

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace std;

unsigned filter_sources_by_flux(unsigned * num_sources,
        const double min_brightness, const double max_brightness,
        double * ra, double *dec, double * brightness)
{
    // Sort the source brightness into increasing order, holding onto
    // the positions of the original indices.
    std::vector<unsigned> indices(*num_sources);
    unsigned * idx = &indices[0];
    LinkedSort::sortIndices(*num_sources, brightness, idx);

    // Find the indices of the sorted radial distance distance array
    // corresponding to the inner and outer radius.
    unsigned iStart = 0;
    for (unsigned i = 0; i < *num_sources; ++i)
    {
        if (brightness[i] >= min_brightness)
        {
            iStart = i;
            break;
        }
    }
    unsigned iEnd = 0;
    for (unsigned i = *num_sources-1;(i >= iStart && i < *num_sources); --i)
    {
        if (brightness[i] <= max_brightness)
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

