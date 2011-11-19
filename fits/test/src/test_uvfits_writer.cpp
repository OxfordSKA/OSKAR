#include "fits/oskar_uvfits_writer.h"

#include <cstdio>
#include <fitsio.h>

int main(int /*argc*/, char** /*argv*/)
{
    // Open the file.
    const char* filename = "test.fits";
    oskar_uvfits uvfits;
    oskar_uvfits_create(filename, &uvfits);

    // Groups header.
    long long num_vis = 2;
    oskar_uvfits_write_groups_header(uvfits.fptr, num_vis);

    // Close file.
    oskar_uvfits_close(uvfits.fptr);

    return 0;
}
