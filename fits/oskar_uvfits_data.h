#ifndef OSKAR_UVFITS_DATA_H_
#define OSKAR_UVFITS_DATA_H_

#include <fitsio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_uvfits
{
    fitsfile* fptr;
    int       status;
    int       decimals;
    int       num_axes;
    int       num_param;
};
typedef struct oskar_uvfits oskar_uvfits;


enum oskar_uvfits_axis_type
{ GROUPS_NONE = 0, AMP = 1, STOKES = 2, FREQ = 3, RA = 4, DEC = 5 };

struct oskar_uvfits_header
{
    long  num_axes;
    long* axis_dim;

    long long num_param; // == pcount
    long long gcount;    // == num_vis
};



#ifdef __cplusplus
}
#endif
#endif // OSKAR_UVFITS_DATA_H_
