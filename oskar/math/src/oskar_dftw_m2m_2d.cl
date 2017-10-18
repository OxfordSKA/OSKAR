/* Copyright (c) 2017, The University of Oxford. See LICENSE file. */

kernel void dftw_m2m_2d_REAL(const int num_in,
        const REAL wavenumber,
        global const REAL* restrict x_in,
        global const REAL* restrict y_in,
        global const REAL2* restrict weights_in,
        const int num_out,
        global const REAL* restrict x_out,
        global const REAL* restrict y_out,
        global const REAL8* restrict data,
        global REAL8* restrict output,
        const int max_in_chunk,
        local REAL2* restrict c_w,
        local REAL2* restrict c_xy)
{
    const int block_dim = get_local_size(0);
    const int thread_idx = get_local_id(0);
    const int i_out = get_global_id(0);
    REAL8 out = (REAL8)(0., 0., 0., 0., 0., 0., 0., 0.);
    REAL xo = (REAL) 0., yo = (REAL) 0.;
    if (i_out < num_out) {
        xo = wavenumber * x_out[i_out];
        yo = wavenumber * y_out[i_out];
    }
    for (int j = 0; j < num_in; j += max_in_chunk) {
        int chunk_size = num_in - j;
        if (chunk_size > max_in_chunk) chunk_size = max_in_chunk;
        // Using block_dim threads, cache chunk_size items of data.
        for (int t = thread_idx; t < chunk_size; t += block_dim) {
            const int g = j + t; // Global input index.
            c_w[t] = weights_in[g];
            c_xy[t].x = x_in[g];
            c_xy[t].y = y_in[g];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i_out < num_out) {
            for (int i = 0; i < chunk_size; ++i) {
                REAL re, im, t;
                REAL2 w = c_w[i];
                t = xo * c_xy[i].x + yo * c_xy[i].y;
                im = sincos(t, &re);
                t = re;
                re *= w.x;
                re -= w.y * im;
                im *= w.x;
                im += w.y * t;
                REAL8 in = data[(j + i) * num_out + i_out];
                out.s0 += in.s0 * re; out.s0 -= in.s1 * im;
                out.s1 += in.s1 * re; out.s1 += in.s0 * im;
                out.s2 += in.s2 * re; out.s2 -= in.s3 * im;
                out.s3 += in.s3 * re; out.s3 += in.s2 * im;
                out.s4 += in.s4 * re; out.s4 -= in.s5 * im;
                out.s5 += in.s5 * re; out.s5 += in.s4 * im;
                out.s6 += in.s6 * re; out.s6 -= in.s7 * im;
                out.s7 += in.s7 * re; out.s7 += in.s6 * im;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i_out < num_out) output[i_out] = out;
}

kernel void dftw_m2m_2d_cpu_REAL(const int num_in,
        const REAL wavenumber,
        global const REAL* restrict x_in,
        global const REAL* restrict y_in,
        global const REAL2* restrict weights_in,
        const int num_out,
        global const REAL* restrict x_out,
        global const REAL* restrict y_out,
        global const REAL8* restrict data,
        global REAL8* restrict output)
{
    const int i_out = get_global_id(0);
    if (i_out >= num_out) return;
    const REAL xo = wavenumber * x_out[i_out];
    const REAL yo = wavenumber * y_out[i_out];
    REAL8 out = (REAL8)(0., 0., 0., 0., 0., 0., 0., 0.);
    for (int i = 0; i < num_in; ++i) {
        REAL re, im, t;
        REAL2 w = weights_in[i];
        t = xo * x_in[i] + yo * y_in[i];
        im = sincos(t, &re);
        t = re;
        re *= w.x;
        re -= w.y * im;
        im *= w.x;
        im += w.y * t;
        REAL8 in = data[i * num_out + i_out];
        out.s0 += in.s0 * re; out.s0 -= in.s1 * im;
        out.s1 += in.s1 * re; out.s1 += in.s0 * im;
        out.s2 += in.s2 * re; out.s2 -= in.s3 * im;
        out.s3 += in.s3 * re; out.s3 += in.s2 * im;
        out.s4 += in.s4 * re; out.s4 -= in.s5 * im;
        out.s5 += in.s5 * re; out.s5 += in.s4 * im;
        out.s6 += in.s6 * re; out.s6 -= in.s7 * im;
        out.s7 += in.s7 * re; out.s7 += in.s6 * im;
    }
    output[i_out] = out;
}
