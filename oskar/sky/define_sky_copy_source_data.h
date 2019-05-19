/* Copyright (c) 2017-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_SKY_COPY_SOURCE_DATA(NAME, FP) KERNEL(NAME) (const int num,\
        GLOBAL_IN(int, mask),    GLOBAL_IN(int, indices),\
        GLOBAL const FP* ra_in,   GLOBAL FP* ra_out,\
        GLOBAL const FP* dec_in,  GLOBAL FP* dec_out,\
        GLOBAL const FP* I_in,    GLOBAL FP* I_out,\
        GLOBAL const FP* Q_in,    GLOBAL FP* Q_out,\
        GLOBAL const FP* U_in,    GLOBAL FP* U_out,\
        GLOBAL const FP* V_in,    GLOBAL FP* V_out,\
        GLOBAL const FP* ref_in,  GLOBAL FP* ref_out,\
        GLOBAL const FP* sp_in,   GLOBAL FP* sp_out,\
        GLOBAL const FP* rm_in,   GLOBAL FP* rm_out,\
        GLOBAL const FP* l_in,    GLOBAL FP* l_out,\
        GLOBAL const FP* m_in,    GLOBAL FP* m_out,\
        GLOBAL const FP* n_in,    GLOBAL FP* n_out,\
        GLOBAL const FP* a_in,    GLOBAL FP* a_out,\
        GLOBAL const FP* b_in,    GLOBAL FP* b_out,\
        GLOBAL const FP* c_in,    GLOBAL FP* c_out,\
        GLOBAL const FP* maj_in,  GLOBAL FP* maj_out,\
        GLOBAL const FP* min_in,  GLOBAL FP* min_out,\
        GLOBAL const FP* pa_in,   GLOBAL FP* pa_out\
)\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    if (mask[i]) {\
        const int i_out = indices[i];\
        ra_out[i_out]  = ra_in[i];\
        dec_out[i_out] = dec_in[i];\
        I_out[i_out]   = I_in[i];\
        Q_out[i_out]   = Q_in[i];\
        U_out[i_out]   = U_in[i];\
        V_out[i_out]   = V_in[i];\
        ref_out[i_out] = ref_in[i];\
        sp_out[i_out]  = sp_in[i];\
        rm_out[i_out]  = rm_in[i];\
        l_out[i_out]   = l_in[i];\
        m_out[i_out]   = m_in[i];\
        n_out[i_out]   = n_in[i];\
        a_out[i_out]   = a_in[i];\
        b_out[i_out]   = b_in[i];\
        c_out[i_out]   = c_in[i];\
        maj_out[i_out] = maj_in[i];\
        min_out[i_out] = min_in[i];\
        pa_out[i_out]  = pa_in[i];\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
