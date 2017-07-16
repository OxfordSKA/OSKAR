#ifndef OSKAR_PRIVATE_RANDOM_GENERATORS_H_
#define OSKAR_PRIVATE_RANDOM_GENERATORS_H_

#include <Random123/philox.h>

/* Use 32-bit integers for both single and double floating-point precision to
 * preserve random sequences. */
/* Generate two random integers. */
#define OSKAR_R123_GENERATE_2(S,C0,C1) \
        philox2x32_key_t k;     \
        philox2x32_ctr_t c;     \
        union {                 \
            philox2x32_ctr_t c; \
            uint32_t i[2];      \
        } u;                    \
        k.v[0] = S;             \
        c.v[0] = C0;            \
        c.v[1] = C1;            \
        u.c = philox2x32(c, k);

/* Generate four random integers. */
#define OSKAR_R123_GENERATE_4(S,C0,C1,C2,C3) \
        philox4x32_key_t k;     \
        philox4x32_ctr_t c;     \
        union {                 \
            philox4x32_ctr_t c; \
            uint32_t i[4];      \
        } u;                    \
        k.v[0] = S;             \
        k.v[1] = 0xCAFEF00DuL;  \
        c.v[0] = C0;            \
        c.v[1] = C1;            \
        c.v[2] = C2;            \
        c.v[3] = C3;            \
        u.c = philox4x32(c, k);

#endif
