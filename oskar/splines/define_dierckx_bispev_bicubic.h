/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

/* Evaluates the (k+1) non-zero bicubic b-splines
 * at t(l) <= x < t(l+1) using the stable recurrence
 * relation of de Boor and Cox. */
#define FPBSPL(FP, t, k, x, l, h)\
    h[0] = (FP)1;\
    for (int j = 1; j <= k; ++j) {\
        for (int i = 0; i < j; ++i) hh[i] = h[i];\
        h[0] = (FP)0;\
        for (int i = 0; i < j; ++i) {\
            const int li = l + i;\
            const int lj = li - j;\
            const FP f = hh[i] / (t[li] - t[lj]);\
            h[i] += f * (t[li] - x);\
            h[i + 1] = f * (x - t[lj]);\
        }\
    }\

#define OSKAR_DIERCKX_BISPEV_BICUBIC(NAME, FP) KERNEL(NAME) (\
        GLOBAL_IN(FP, tx), const int nx, GLOBAL_IN(FP, ty), const int ny,\
        GLOBAL_IN(FP, c), const int n, GLOBAL_IN(FP, x), GLOBAL_IN(FP, y),\
        const int stride_out, const int offset_out, GLOBAL_OUT(FP, z))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    int l, l1, l2, nk1, lx;\
    FP hh[3], wx[4], wy[4], t, x_ = x[i], y_ = y[i];\
    nk1 = nx - 4;\
    t = tx[3];   if (x_ < t) x_ = t;\
    t = tx[nk1]; if (x_ > t) x_ = t;\
    l = 4; while (!(x_ < tx[l] || l == nk1)) l++;\
    FPBSPL(FP, tx, 3, x_, l, wx)\
    lx = l - 4;\
    nk1 = ny - 4;\
    t = ty[3];   if (y_ < t) y_ = t;\
    t = ty[nk1]; if (y_ > t) y_ = t;\
    l = 4; while (!(y_ < ty[l] || l == nk1)) l++;\
    FPBSPL(FP, ty, 3, y_, l, wy)\
    l1 = lx * nk1 + (l - 4);\
    t = (FP)0;\
    for (l = 0; l <= 3; ++l) {\
        l2 = l1;\
        for (int j = 0; j <= 3; ++j, ++l2) t += c[l2] * wx[l] * wy[j];\
        l1 += nk1;\
    }\
    z[i * stride_out + offset_out] = t;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_SET_ZEROS_STRIDE(NAME, FP) KERNEL(NAME) (const int n,\
        const int stride_out, const int offset_out, GLOBAL_OUT(FP, out))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    out[i * stride_out + offset_out] = (FP)0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
