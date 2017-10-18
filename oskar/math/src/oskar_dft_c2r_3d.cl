/* Copyright (c) 2017, The University of Oxford. See LICENSE file. */

kernel void dft_c2r_3d_REAL(const int num_in,
        const REAL wavenumber,
        global const REAL* restrict x_in,
        global const REAL* restrict y_in,
        global const REAL* restrict z_in,
        global const REAL2* restrict data_in,
        global const REAL* restrict weight_in,
        const int num_out,
        global const REAL* restrict x_out,
        global const REAL* restrict y_out,
        global const REAL* restrict z_out,
        global REAL* restrict output,
        const int max_in_chunk,
        local REAL2* restrict c_d,
        local REAL2* restrict c_xy,
        local REAL* restrict c_z)
{
    const int block_dim = get_local_size(0);
    const int thread_idx = get_local_id(0);
    const int i_out = get_global_id(0);
    REAL out = (REAL) 0.;
    REAL xo = (REAL) 0., yo = (REAL) 0., zo = (REAL) 0.;
    if (i_out < num_out) {
        xo = wavenumber * x_out[i_out];
        yo = wavenumber * y_out[i_out];
        zo = wavenumber * z_out[i_out];
    }
    for (int j = 0; j < num_in; j += max_in_chunk) {
        int chunk_size = num_in - j;
        if (chunk_size > max_in_chunk) chunk_size = max_in_chunk;
        // Using block_dim threads, cache chunk_size items of data.
        for (int t = thread_idx; t < chunk_size; t += block_dim) {
            const int g = j + t; // Global input index.
            c_d[t] = data_in[g] * weight_in[g];
            c_xy[t].x = x_in[g];
            c_xy[t].y = y_in[g];
            c_z[t] = z_in[g];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < chunk_size; ++i) {
            const REAL2 d = c_d[i];
            const REAL phase = xo * c_xy[i].x + yo * c_xy[i].y + zo * c_z[i];
            REAL re, im;
            im = sincos(-phase, &re);
            out += d.x * re;
            out -= d.y * im;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i_out < num_out) output[i_out] = out;
}

kernel void dft_c2r_3d_cpu_REAL(const int num_in,
        const REAL wavenumber,
        global const REAL* restrict x_in,
        global const REAL* restrict y_in,
        global const REAL* restrict z_in,
        global const REAL2* restrict data_in,
        global const REAL* restrict weight_in,
        const int num_out,
        global const REAL* restrict x_out,
        global const REAL* restrict y_out,
        global const REAL* restrict z_out,
        global REAL* restrict output)
{
    const int i_out = get_global_id(0);
    if (i_out >= num_out) return;
    const REAL xo = wavenumber * x_out[i_out];
    const REAL yo = wavenumber * y_out[i_out];
    const REAL zo = wavenumber * z_out[i_out];
    REAL out = (REAL) 0.;
    for (int i = 0; i < num_in; ++i) {
        const REAL2 d = data_in[i] * weight_in[i];
        const REAL phase = xo * x_in[i] + yo * y_in[i] + zo * z_in[i];
        REAL re, im;
        im = sincos(-phase, &re);
        out += d.x * re;
        out -= d.y * im;
    }
    output[i_out] = out;
}
