/* Copyright (c) 2011-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_XCORR_ARGS(FP)\
        const int num_src, const int num_stations, const int offset_out,\
        GLOBAL_IN(FP, src_I), GLOBAL_IN(FP, src_Q),\
        GLOBAL_IN(FP, src_U), GLOBAL_IN(FP, src_V),\
        GLOBAL_IN(FP, src_l), GLOBAL_IN(FP, src_m), GLOBAL_IN(FP, src_n),\
        GLOBAL_IN(FP, src_a), GLOBAL_IN(FP, src_b), GLOBAL_IN(FP, src_c),\
        GLOBAL_IN(FP, st_u), GLOBAL_IN(FP, st_v), GLOBAL_IN(FP, st_w),\
        GLOBAL_IN(FP, st_x), GLOBAL_IN(FP, st_y),\
        const FP uv_min_lambda,  const FP uv_max_lambda,\
        const FP inv_wavelength, const FP frac_bandwidth,\
        const FP time_int_sec,   const FP gha0_rad, const FP dec0_rad,\

#define OSKAR_XCORR_SMEARING(BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP)\
        FP smearing;\
        if (GAUSSIAN) {\
            const FP t = -(src_a[i] * uu2 + src_b[i] * uuvv + src_c[i] * vv2);\
            smearing = exp(t);\
        } else smearing = (FP) 1;\
        if (BANDWIDTH_SMEARING || TIME_SMEARING) {\
            const FP l = src_l[i], m = src_m[i], n = src_n[i] - (FP) 1;\
            if (BANDWIDTH_SMEARING) {\
                const FP t = uu * l + vv * m + ww * n;\
                smearing *= OSKAR_SINC(FP, t);\
            }\
            if (TIME_SMEARING) {\
                const FP t = du * l + dv * m + dw * n;\
                smearing *= OSKAR_SINC(FP, t);\
            }\
        }\

#define OSKAR_XCORR_GPU(NAME, BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP, FP2, FP4c)\
KERNEL(NAME) (OSKAR_XCORR_ARGS(FP)\
        GLOBAL_IN(FP4c, jones), GLOBAL_OUT(FP4c, vis) LOCAL_CL(FP4c, smem))\
{\
    const int SP = GROUP_ID_X, SQ = GROUP_ID_Y;\
    const int bdim = LOCAL_DIM_X, tid = LOCAL_ID_X;\
    LOCAL FP uv_len, uu, vv, ww, uu2, vv2, uuvv, du, dv, dw;\
    FP4c m1, m2, sum;\
    LOCAL_CUDA_BASE(FP4c, smem)\
    if (SQ >= SP) return;\
    if (tid == 0) {\
        OSKAR_BASELINE_TERMS(FP, st_u[SP], st_u[SQ], st_v[SP], st_v[SQ],\
                st_w[SP], st_w[SQ], uu, vv, ww, uu2, vv2, uuvv, uv_len)\
        if (TIME_SMEARING)\
            OSKAR_BASELINE_DELTAS(FP, st_x[SP], st_x[SQ],\
                    st_y[SP], st_y[SQ], du, dv, dw)\
    }\
    BARRIER;\
    if (uv_len < uv_min_lambda || uv_len > uv_max_lambda) return;\
    GLOBAL_IN(FP4c, st_p) = &jones[num_src * SP];\
    GLOBAL_IN(FP4c, st_q) = &jones[num_src * SQ];\
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, sum)\
    for (int i = tid; i < num_src; i += bdim) {\
        OSKAR_XCORR_SMEARING(BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP)\
        OSKAR_CONSTRUCT_B(FP, m2, src_I[i], src_Q[i], src_U[i], src_V[i])\
        OSKAR_LOAD_MATRIX(m1, st_p[i])\
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)\
        OSKAR_LOAD_MATRIX(m2, st_q[i])\
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)\
        OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(sum, m1, smearing)\
    }\
    smem[tid] = sum;\
    BARRIER;\
    if (tid == 0) {\
        for (int i = 1; i < bdim; ++i)\
            OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(sum, smem[i])\
        int i = OSKAR_BASELINE_INDEX(num_stations, SP, SQ) + offset_out;\
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[i], sum)\
    }\
}

#define OSKAR_XCORR_CPU(NAME, BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP, FP2, FP4c)\
KERNEL(NAME) (OSKAR_XCORR_ARGS(FP)\
        GLOBAL_IN(FP4c, jones), GLOBAL_OUT(FP4c, vis))\
{\
    const int SP = GLOBAL_ID_X, SQ = GROUP_ID_Y;\
    FP uv_len, uu, vv, ww, uu2, vv2, uuvv, du, dv, dw;\
    FP4c m1, m2, sum;\
    if (SQ >= SP || SP >= num_stations) return;\
    OSKAR_BASELINE_TERMS(FP, st_u[SP], st_u[SQ], st_v[SP], st_v[SQ],\
            st_w[SP], st_w[SQ], uu, vv, ww, uu2, vv2, uuvv, uv_len)\
    if (TIME_SMEARING)\
        OSKAR_BASELINE_DELTAS(FP, st_x[SP], st_x[SQ],\
                st_y[SP], st_y[SQ], du, dv, dw)\
    if (uv_len < uv_min_lambda || uv_len > uv_max_lambda) return;\
    GLOBAL_IN(FP4c, st_p) = &jones[num_src * SP];\
    GLOBAL_IN(FP4c, st_q) = &jones[num_src * SQ];\
    const int j = OSKAR_BASELINE_INDEX(num_stations, SP, SQ) + offset_out;\
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, sum)\
    for (int i = 0; i < num_src; i++) {\
        OSKAR_XCORR_SMEARING(BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP)\
        OSKAR_CONSTRUCT_B(FP, m2, src_I[i], src_Q[i], src_U[i], src_V[i])\
        OSKAR_LOAD_MATRIX(m1, st_p[i])\
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)\
        OSKAR_LOAD_MATRIX(m2, st_q[i])\
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)\
        OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(sum, m1, smearing)\
    }\
    OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[j], sum)\
}

#define OSKAR_XCORR_SCALAR_GPU(NAME, BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP, FP2)\
KERNEL(NAME) (OSKAR_XCORR_ARGS(FP)\
        GLOBAL_IN(FP2, jones), GLOBAL_OUT(FP2, vis) LOCAL_CL(FP2, smem))\
{\
    const int SP = GROUP_ID_X, SQ = GROUP_ID_Y;\
    const int bdim = LOCAL_DIM_X, tid = LOCAL_ID_X;\
    LOCAL FP uv_len, uu, vv, ww, uu2, vv2, uuvv, du, dv, dw;\
    FP2 t1, t2, sum;\
    LOCAL_CUDA_BASE(FP2, smem)\
    if (SQ >= SP) return;\
    if (tid == 0) {\
        OSKAR_BASELINE_TERMS(FP, st_u[SP], st_u[SQ], st_v[SP], st_v[SQ],\
                st_w[SP], st_w[SQ], uu, vv, ww, uu2, vv2, uuvv, uv_len)\
        if (TIME_SMEARING)\
            OSKAR_BASELINE_DELTAS(FP, st_x[SP], st_x[SQ],\
                    st_y[SP], st_y[SQ], du, dv, dw)\
    }\
    BARRIER;\
    if (uv_len < uv_min_lambda || uv_len > uv_max_lambda) return;\
    GLOBAL_IN(FP2, st_p) = &jones[num_src * SP];\
    GLOBAL_IN(FP2, st_q) = &jones[num_src * SQ];\
    MAKE_ZERO2(FP, sum);\
    for (int i = tid; i < num_src; i += bdim) {\
        OSKAR_XCORR_SMEARING(BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP)\
        smearing *= src_I[i];\
        t1 = st_p[i]; t2 = st_q[i];\
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, t1, t2)\
        sum.x += t1.x * smearing; sum.y += t1.y * smearing;\
    }\
    smem[tid] = sum;\
    BARRIER;\
    if (tid == 0) {\
        for (int i = 1; i < bdim; ++i) {\
            sum.x += smem[i].x; sum.y += smem[i].y;\
        }\
        int i = OSKAR_BASELINE_INDEX(num_stations, SP, SQ) + offset_out;\
        vis[i].x += sum.x; vis[i].y += sum.y;\
    }\
}

#define OSKAR_XCORR_SCALAR_CPU(NAME, BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP, FP2)\
KERNEL(NAME) (OSKAR_XCORR_ARGS(FP)\
        GLOBAL_IN(FP2, jones), GLOBAL_OUT(FP2, vis))\
{\
    const int SP = GLOBAL_ID_X, SQ = GROUP_ID_Y;\
    FP uv_len, uu, vv, ww, uu2, vv2, uuvv, du, dv, dw;\
    FP2 t1, t2, sum;\
    if (SQ >= SP || SP >= num_stations) return;\
    OSKAR_BASELINE_TERMS(FP, st_u[SP], st_u[SQ], st_v[SP], st_v[SQ],\
            st_w[SP], st_w[SQ], uu, vv, ww, uu2, vv2, uuvv, uv_len)\
    if (TIME_SMEARING)\
        OSKAR_BASELINE_DELTAS(FP, st_x[SP], st_x[SQ],\
                st_y[SP], st_y[SQ], du, dv, dw)\
    if (uv_len < uv_min_lambda || uv_len > uv_max_lambda) return;\
    GLOBAL_IN(FP2, st_p) = &jones[num_src * SP];\
    GLOBAL_IN(FP2, st_q) = &jones[num_src * SQ];\
    const int j = OSKAR_BASELINE_INDEX(num_stations, SP, SQ) + offset_out;\
    MAKE_ZERO2(FP, sum);\
    for (int i = 0; i < num_src; i++) {\
        OSKAR_XCORR_SMEARING(BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP)\
        smearing *= src_I[i];\
        t1 = st_p[i]; t2 = st_q[i];\
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, t1, t2)\
        sum.x += t1.x * smearing; sum.y += t1.y * smearing;\
    }\
    vis[j].x += sum.x; vis[j].y += sum.y;\
}
