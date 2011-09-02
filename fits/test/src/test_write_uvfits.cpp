#include "fits/oskar_write_uvfits.h"

#include <cstdio>
#include <fitsio.h>

int main(int /*argc*/, char** /*argv*/)
{
    // Open the file.
    const char* filename = "test.fits";
    fitsfile* fits = NULL;
    oskar_open_uvfits_file(filename, &fits);

    // Groups header.
    long long num_vis = 2;
    long num_stokes = 1;
    long num_freqs = 1;
    long num_ra = 1;
    long num_dec = 1;
    oskar_write_groups_header(fits, num_vis, num_stokes, num_freqs, num_ra, num_dec);

    // Close file.
    oskar_close_uvfits_file(fits);

    return 0;
}
