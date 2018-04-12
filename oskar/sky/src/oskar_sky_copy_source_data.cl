kernel void copy_source_data_REAL(const int num,
        global const int* restrict mask, global const int* restrict indices,
        global const REAL* ra_in,   global REAL* ra_out,
        global const REAL* dec_in,  global REAL* dec_out,
        global const REAL* I_in,    global REAL* I_out,
        global const REAL* Q_in,    global REAL* Q_out,
        global const REAL* U_in,    global REAL* U_out,
        global const REAL* V_in,    global REAL* V_out,
        global const REAL* ref_in,  global REAL* ref_out,
        global const REAL* sp_in,   global REAL* sp_out,
        global const REAL* rm_in,   global REAL* rm_out,
        global const REAL* l_in,    global REAL* l_out,
        global const REAL* m_in,    global REAL* m_out,
        global const REAL* n_in,    global REAL* n_out,
        global const REAL* a_in,    global REAL* a_out,
        global const REAL* b_in,    global REAL* b_out,
        global const REAL* c_in,    global REAL* c_out,
        global const REAL* maj_in,  global REAL* maj_out,
        global const REAL* min_in,  global REAL* min_out,
        global const REAL* pa_in,   global REAL* pa_out
        )
{
    const int i = get_global_id(0);
    if (i >= num) return;
    if (mask[i])
    {
        const int i_out = indices[i];
        ra_out[i_out]  = ra_in[i];
        dec_out[i_out] = dec_in[i];
        I_out[i_out]   = I_in[i];
        Q_out[i_out]   = Q_in[i];
        U_out[i_out]   = U_in[i];
        V_out[i_out]   = V_in[i];
        ref_out[i_out] = ref_in[i];
        sp_out[i_out]  = sp_in[i];
        rm_out[i_out]  = rm_in[i];
        l_out[i_out]   = l_in[i];
        m_out[i_out]   = m_in[i];
        n_out[i_out]   = n_in[i];
        a_out[i_out]   = a_in[i];
        b_out[i_out]   = b_in[i];
        c_out[i_out]   = c_in[i];
        maj_out[i_out] = maj_in[i];
        min_out[i_out] = min_in[i];
        pa_out[i_out]  = pa_in[i];
    }
}
