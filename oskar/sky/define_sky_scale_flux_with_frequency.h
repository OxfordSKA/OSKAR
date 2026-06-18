/* Copyright (c) 2014-2026, The OSKAR Developers. See LICENSE file. */

#define C_0 299792458

#define OSKAR_SKY_CALC_STOKES_I_FLUX(NAME, FP) DEVICE_FUNC FP NAME (\
        const int capacity,\
        GLOBAL_IN(FP, table),\
        GLOBAL_IN(int, num_valid_col),\
        const int i_src,\
        const int col_i,\
        const int col_ref_freq,\
        const int col_lin_sp_index,\
        const int col_sp_index,\
        const int col_sp_curvature,\
        const int col_line_width,\
        const int col_freq_inc,\
        const FP stokes_i_in,\
        const FP freq_hz,\
        const FP freq0_hz\
)\
{\
    FP lin_spx = (FP) 0;\
    int c = 0;\
    const int num_stokes_i = num_valid_col[OSKAR_SKY_I_JY];\
    const int num_line_width_hz = num_valid_col[OSKAR_SKY_LINE_WIDTH_HZ];\
    if (num_line_width_hz > 0) {\
        /* Spectral line source. Calculate flux using Gaussian profile(s). */\
        FP stokes_i_out = (FP) 0;\
        FP sigma = (FP) 0, inc_hz = (FP) 0, ref_hz = (FP) 0;\
        const int num_ref_freq = num_valid_col[OSKAR_SKY_REF_HZ];\
        const int num_freq_inc = num_valid_col[OSKAR_SKY_INC_HZ];\
        if (num_freq_inc > 0) inc_hz = table[capacity * col_freq_inc + i_src];\
        for (c = 0; c < num_stokes_i; ++c) {\
            const FP amp = table[capacity * (col_i + c) + i_src];\
            if (c < num_ref_freq) {\
                ref_hz = table[capacity * (col_ref_freq + c) + i_src];\
            } else {\
                ref_hz = table[capacity * col_ref_freq + i_src] + c * inc_hz;\
            }\
            if (c < num_line_width_hz) {\
                sigma = table[capacity * (col_line_width + c) + i_src];\
            } else {\
                sigma = table[capacity * col_line_width + i_src];\
            }\
            const FP x = freq_hz - ref_hz;\
            stokes_i_out += amp * exp(-(x * x) / (2 * sigma * sigma));\
        }\
        return stokes_i_out;\
    }\
    /* Use of any spectral index requires only a single Stokes I value. */\
    if (num_stokes_i > 1) return stokes_i_in;\
    const int num_sp_indices = num_valid_col[OSKAR_SKY_SPEC_IDX];\
    if (num_valid_col[OSKAR_SKY_SPEC_CURV] > 0) {\
        /* Follow Equation 2 of Callingham et al. 2017. */\
        FP sp_index = (FP) 0;\
        FP sp_curvature = table[capacity * col_sp_curvature + i_src];\
        if (num_sp_indices > 0) {\
            sp_index = table[capacity * col_sp_index + i_src];\
        }\
        if (sp_index != sp_index) sp_index = (FP) 0;\
        if (sp_curvature != sp_curvature) sp_curvature = (FP) 0;\
        const FP freq_ratio = freq_hz / freq0_hz;\
        const FP curv = exp(sp_curvature * pow(log(freq_ratio), 2));\
        return stokes_i_in * curv * pow(freq_ratio, sp_index);\
    }\
    const int have_lin_si = num_valid_col[OSKAR_SKY_LIN_SI] > 0;\
    if (have_lin_si) lin_spx = table[capacity * col_lin_sp_index + i_src];\
    if (lin_spx == (FP) 0) {\
        /* Default is logarithmic spectral index. */\
        FP exponent = (FP) 0;\
        const FP base = log10(freq_hz / freq0_hz);\
        /* Use Horner's method to evaluate spectral index polynomial. */\
        /* Unfortunately sequential, but it avoids many calls to pow(). */\
        for (c = num_sp_indices - 1; c >= 0; --c) {\
            FP sp_index = table[capacity * (col_sp_index + c) + i_src];\
            if (sp_index != sp_index) sp_index = (FP) 0;\
            exponent = base * exponent + sp_index;\
        }\
        return stokes_i_in * pow((FP) 10, base * exponent);\
    }\
    /* Linear spectral index, used (only?) by WSClean. */\
    FP value = (FP) 0;\
    const FP base = (freq_hz / freq0_hz) - (FP) 1;\
    for (c = num_sp_indices - 1; c >= 0; --c) {\
        FP sp_index = table[capacity * (col_sp_index + c) + i_src];\
        if (sp_index != sp_index) sp_index = (FP) 0;\
        value = base * value + sp_index;\
    }\
    return stokes_i_in + base * value;\
}\


#define OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY(NAME, FP) KERNEL(NAME) (\
        const int num_sources,\
        const int capacity,\
        const FP freq_hz,\
        GLOBAL_IN(FP, table),\
        GLOBAL_IN(int, num_valid_columns),\
        const int col_i,\
        const int col_q,\
        const int col_u,\
        const int col_v,\
        const int col_ref_freq,\
        const int col_lin_sp_index,\
        const int col_sp_index,\
        const int col_rot_meas,\
        const int col_pol_frac,\
        const int col_pol_ang,\
        const int col_ref_wavelength,\
        const int col_sp_curvature,\
        const int col_line_width,\
        const int col_freq_inc,\
        GLOBAL_OUT(FP, out_i),\
        GLOBAL_OUT(FP, out_q),\
        GLOBAL_OUT(FP, out_u),\
        GLOBAL_OUT(FP, out_v)\
)\
{\
    const int col_iquv[4] = {col_i, col_q, col_u, col_v};\
    KERNEL_LOOP_X(int, i, 0, num_sources)\
    FP in_iquv[4] = {(FP) 0, (FP) 0, (FP) 0, (FP) 0};\
    int num_iquv[4] = {0, 0, 0, 0};\
    int j = 0, chan_idx = 0;\
    GLOBAL_IN(int, num_valid_col) = &num_valid_columns[\
            i * OSKAR_SKY_NUM_FIXED_COLUMN_TYPES\
    ];\
    /* Find the sky model channel index closest to the current frequency. */\
    for (j = 0; j < 4; ++j) {\
        num_iquv[j] = num_valid_col[(int) OSKAR_SKY_I_JY + j];\
    }\
    const int num_ref_freq = num_valid_col[OSKAR_SKY_REF_HZ];\
    const int num_freq_inc = num_valid_col[OSKAR_SKY_INC_HZ];\
    FP diff = 1e38, freq0_hz = (FP) 0;\
    if (num_freq_inc == 1 && num_ref_freq == 1) {\
        const FP start_freq_hz = table[capacity * col_ref_freq + i];\
        const FP freq_inc_hz = table[capacity * col_freq_inc + i];\
        for (j = 0; j < num_iquv[0]; ++j) {\
            const FP test_freq_hz = start_freq_hz + j * freq_inc_hz;\
            const FP temp = fabs(test_freq_hz - freq_hz);\
            if (temp < diff) {\
                diff = temp;\
                chan_idx = j;\
                freq0_hz = test_freq_hz;\
            }\
        }\
    }\
    else {\
        for (j = 0; j < num_iquv[0]; ++j) {\
            const FP test_freq_hz = table[capacity * (col_ref_freq + j) + i];\
            const FP temp = fabs(test_freq_hz - freq_hz);\
            if (temp < diff) {\
                diff = temp;\
                chan_idx = j;\
                freq0_hz = test_freq_hz;\
            }\
        }\
    }\
    /* Get reference Stokes parameters. */\
    for (j = 0; j < 4; ++j) {\
        if (num_iquv[j] > 0) {\
            if (chan_idx < num_iquv[j]) {\
                in_iquv[j] = table[capacity * (col_iquv[j] + chan_idx) + i];\
            } else {\
                in_iquv[j] = table[capacity * col_iquv[j] + i];\
            }\
        }\
        if (in_iquv[j] != in_iquv[j]) in_iquv[j] = (FP) 0; /* Catch NaN. */\
    }\
    /* Calculate reference Q and U if possible, and if not supplied. */\
    const int num_pol_angle = num_valid_col[OSKAR_SKY_POLA_RAD];\
    const int num_pol_frac = num_valid_col[OSKAR_SKY_POLF];\
    const int have_rotation_measure = num_valid_col[OSKAR_SKY_RM_RAD] > 0;\
    const int have_ref_wavelength = num_valid_col[OSKAR_SKY_REF_WAVE_M] > 0;\
    if (num_pol_angle > 0 && num_pol_frac > 0 && (\
            num_iquv[1] <= 0 || num_iquv[2] <= 0)) {\
        FP sin_angle = (FP) 0, cos_angle = (FP) 0;\
        FP pol_frac = (FP) 0, pol_ang_rad = (FP) 0;\
        if (chan_idx < num_pol_angle) {\
            pol_ang_rad = table[capacity * (col_pol_ang + chan_idx) + i];\
        } else {\
            pol_ang_rad = table[capacity * col_pol_ang + i];\
        }\
        if (chan_idx < num_pol_frac) {\
            pol_frac = table[capacity * (col_pol_frac + chan_idx) + i];\
        } else {\
            pol_frac = table[capacity * col_pol_frac + i];\
        }\
        SINCOS((2 * pol_ang_rad), sin_angle, cos_angle);\
        in_iquv[1] = pol_frac * in_iquv[0] * cos_angle; /* Q */\
        in_iquv[2] = pol_frac * in_iquv[0] * sin_angle; /* U */\
    }\
    if (freq0_hz == (FP) 0) {\
        /* If reference frequency is 0, we can't do anything else here. */\
        out_i[i] = in_iquv[0];\
        out_q[i] = in_iquv[1];\
        out_u[i] = in_iquv[2];\
        out_v[i] = in_iquv[3];\
    }\
    else {\
        /* Calculate new Stokes I value based on spectral parameters. */\
        const FP stokes_i_out = M_CAT(calc_stokes_i_flux_, FP)(\
                capacity, table, num_valid_col, i,\
                col_i, col_ref_freq, col_lin_sp_index, col_sp_index,\
                col_sp_curvature, col_line_width, col_freq_inc,\
                in_iquv[0], freq_hz, freq0_hz\
        );\
        /* Find ratio between input and output Stokes I */\
        /* to determine default scaling for other Stokes parameters. */\
        const FP scale = (\
                (in_iquv[0] != (FP) 0) ? stokes_i_out / in_iquv[0] : (FP) 0\
        );\
        /* Deal with the rotation measure, if we have it. */\
        if (have_rotation_measure) {\
            FP sin_two_chi = (FP) 0, cos_two_chi = (FP) 0;\
            FP pol_frac = (FP) 0, pol_ang_chi0 = (FP) 0;\
            FP rm = table[capacity * col_rot_meas + i];\
            if (rm != rm) rm = (FP) 0;\
            const FP wavelength_m = ((FP) C_0) / freq_hz;\
            if (num_pol_angle <= 0 || num_pol_frac <= 0) {\
                /* Compute polarisation angle and fraction if not set. */\
                /* Get reference wavelength for polarisation calculation. */\
                FP ref_wave_m = (FP) 0;\
                if (have_ref_wavelength) {\
                    ref_wave_m = table[capacity * col_ref_wavelength + i];\
                } else {\
                    ref_wave_m = ((FP) C_0) / freq0_hz;\
                }\
                const FP freq_at_lambda0_hz = ((FP) C_0) / ref_wave_m;\
                const FP stokes_i_ref = M_CAT(calc_stokes_i_flux_, FP)(\
                        capacity, table, num_valid_col, i, col_i,\
                        col_ref_freq, col_lin_sp_index, col_sp_index,\
                        col_sp_curvature, col_line_width, col_freq_inc,\
                        in_iquv[0], freq_hz, freq_at_lambda0_hz\
                );\
                pol_ang_chi0 = ((FP) 0.5) * atan2(in_iquv[2], in_iquv[1]) - (\
                        ref_wave_m * ref_wave_m * rm\
                );\
                pol_frac = sqrt(in_iquv[1] * in_iquv[1] + \
                        in_iquv[2] * in_iquv[2]\
                ) / stokes_i_ref;\
            }\
            else {\
                if (chan_idx < num_pol_angle) {\
                    pol_ang_chi0 = table[capacity * (col_pol_ang + chan_idx) + i];\
                } else {\
                    pol_ang_chi0 = table[capacity * col_pol_ang + i];\
                }\
                if (chan_idx < num_pol_frac) {\
                    pol_frac = table[capacity * (col_pol_frac + chan_idx) + i];\
                } else {\
                    pol_frac = table[capacity * col_pol_frac + i];\
                }\
            }\
            const FP chi = pol_ang_chi0 + rm * (wavelength_m * wavelength_m);\
            SINCOS((2 * chi), sin_two_chi, cos_two_chi);\
            out_q[i] = pol_frac * stokes_i_out * cos_two_chi;\
            out_u[i] = pol_frac * stokes_i_out * sin_two_chi;\
        }\
        else {\
            /* No rotation measure: scale Q and U the same way as Stokes I. */\
            out_q[i] = scale * in_iquv[1];\
            out_u[i] = scale * in_iquv[2];\
        }\
        out_i[i] = stokes_i_out;\
        out_v[i] = scale * in_iquv[3];\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
