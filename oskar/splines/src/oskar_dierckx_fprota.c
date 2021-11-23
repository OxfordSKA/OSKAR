#include "splines/oskar_dierckx_fprota.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_dierckx_fprota(double cos_, double sin_, double *a, double *b)
{
    double stor1 = 0.0, stor2 = 0.0;

    stor1 = *a;
    stor2 = *b;
    *b = cos_ * stor2 + sin_ * stor1;
    *a = cos_ * stor1 - sin_ * stor2;
}

#ifdef __cplusplus
}
#endif
