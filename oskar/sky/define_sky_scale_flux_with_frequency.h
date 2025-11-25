/* Copyright (c) 2014-2025, The OSKAR Developers. See LICENSE file. */

#define C_0 299792458

#define OSKAR_SKY_CALC_STOKES_I_FLUX(NAME, FP) DEVICE_FUNC FP NAME (\
        const FP stokes_i_in,\
        const FP freq_hz,\
        const FP freq0_hz,\
        const FP lin_spx,\
        const int num_sp_indices,\
        const FP spxn[8],\
        const int have_sp_curvature,\
        const FP sp_curvature\
)\
{\
    int c = 0;\
    if (have_sp_curvature) {\
        /* Follow Equation 2 of Callingham et al. 2017. */\
        const FP freq_ratio = freq_hz / freq0_hz;\
        const FP curv = exp(sp_curvature * pow(log(freq_ratio), 2));\
        return stokes_i_in * curv * pow(freq_ratio, spxn[0]);\
    }\
    if (lin_spx == (FP) 0) {\
        /* Default is logarithmic spectral index. */\
        FP exponent = (FP) 0;\
        const FP base = log10(freq_hz / freq0_hz);\
        /* Use Horner's method to evaluate spectral index polynomial. */\
        /* Unfortunately sequential, but it avoids many calls to pow(). */\
        for (c = num_sp_indices - 1; c >= 0; --c) {\
            exponent = base * exponent + spxn[c];\
        }\
        return stokes_i_in * pow((FP) 10, base * exponent);\
    }\
    /* Linear spectral index, used (only?) by WSClean. */\
    FP value = (FP) 0;\
    const FP base = (freq_hz / freq0_hz) - (FP) 1;\
    for (c = num_sp_indices - 1; c >= 0; --c) {\
        value = base * value + spxn[c];\
    }\
    return stokes_i_in + base * value;\
}\


#define OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY(NAME, FP) KERNEL(NAME) (\
        const int num_sources,\
        const FP freq_hz,\
        GLOBAL_IN(FP, in_i),\
        GLOBAL_IN(FP, in_q),\
        GLOBAL_IN(FP, in_u),\
        GLOBAL_IN(FP, in_v),\
        GLOBAL_IN(FP, ref_freq_hz),\
        GLOBAL_IN(FP, linear_sp_index),\
        const int num_sp_indices,\
        GLOBAL_IN(FP, sp_index0),\
        GLOBAL_IN(FP, sp_index1),\
        GLOBAL_IN(FP, sp_index2),\
        GLOBAL_IN(FP, sp_index3),\
        GLOBAL_IN(FP, sp_index4),\
        GLOBAL_IN(FP, sp_index5),\
        GLOBAL_IN(FP, sp_index6),\
        GLOBAL_IN(FP, sp_index7),\
        GLOBAL_IN(FP, rotation_measure_rad),\
        GLOBAL_IN(FP, pol_fraction),\
        GLOBAL_IN(FP, pol_angle_rad),\
        GLOBAL_IN(FP, ref_wavelength_m),\
        GLOBAL_IN(FP, sp_curvature),\
        GLOBAL_OUT(FP, out_i),\
        GLOBAL_OUT(FP, out_q),\
        GLOBAL_OUT(FP, out_u),\
        GLOBAL_OUT(FP, out_v)\
)\
{\
    /* Figure out what we have based on the input arguments. */\
    const int have_rotation_measure = rotation_measure_rad ? 1 : 0;\
    const int have_pol_angle = pol_angle_rad ? 1 : 0;\
    const int have_pol_fraction = pol_fraction ? 1 : 0;\
    const int have_ref_wavelength = ref_wavelength_m ? 1 : 0;\
    const int have_sp_curvature = sp_curvature ? 1 : 0;\
    const int have_q = in_q ? 1 : 0;\
    const int have_u = in_u ? 1 : 0;\
    KERNEL_LOOP_X(int, i, 0, num_sources)\
    FP in_iquv[4] = {(FP) 0, (FP) 0, (FP) 0, (FP) 0};\
    const FP freq0_hz = ref_freq_hz ? ref_freq_hz[i] : (FP) 0;\
    if (in_i) in_iquv[0] = in_i[i];\
    if (in_v) in_iquv[3] = in_v[i];\
    if (have_pol_angle && have_pol_fraction && (!have_q || !have_u)) {\
        /* Calculate reference Q and U if possible, and if not supplied. */\
        FP sin_angle = (FP) 0, cos_angle = (FP) 0;\
        SINCOS((2 * pol_angle_rad[i]), sin_angle, cos_angle);\
        in_iquv[1] = pol_fraction[i] * in_iquv[0] * cos_angle; /* Q */\
        in_iquv[2] = pol_fraction[i] * in_iquv[0] * sin_angle; /* U */\
    }\
    else {\
        if (have_q) in_iquv[1] = in_q[i];\
        if (have_u) in_iquv[2] = in_u[i];\
    }\
    /* Catch any NaN values. */\
    if (in_iquv[0] != in_iquv[0]) in_iquv[0] = (FP) 0;\
    if (in_iquv[1] != in_iquv[1]) in_iquv[1] = (FP) 0;\
    if (in_iquv[2] != in_iquv[2]) in_iquv[2] = (FP) 0;\
    if (in_iquv[3] != in_iquv[3]) in_iquv[3] = (FP) 0;\
    if (freq0_hz == (FP) 0) {\
        /* If reference frequency is 0, we can't do anything else here. */\
        out_i[i] = in_iquv[0];\
        out_q[i] = in_iquv[1];\
        out_u[i] = in_iquv[2];\
        out_v[i] = in_iquv[3];\
    }\
    else {\
        FP spxn[8];\
        const FP lin_spx = linear_sp_index ? linear_sp_index[i] : (FP) 0;\
        const FP sp_curv = sp_curvature ? sp_curvature[i] : (FP) 0;\
        spxn[0] = sp_index0 ? sp_index0[i] : (FP) 0;\
        spxn[1] = sp_index1 ? sp_index1[i] : (FP) 0;\
        spxn[2] = sp_index2 ? sp_index2[i] : (FP) 0;\
        spxn[3] = sp_index3 ? sp_index3[i] : (FP) 0;\
        spxn[4] = sp_index4 ? sp_index4[i] : (FP) 0;\
        spxn[5] = sp_index5 ? sp_index5[i] : (FP) 0;\
        spxn[6] = sp_index6 ? sp_index6[i] : (FP) 0;\
        spxn[7] = sp_index7 ? sp_index7[i] : (FP) 0;\
        /* Catch any NaN values. */\
        if (spxn[0] != spxn[0]) spxn[0] = (FP) 0;\
        if (spxn[1] != spxn[1]) spxn[1] = (FP) 0;\
        /* Calculate new Stokes I value based on spectral parameters. */\
        const FP stokes_i_out = M_CAT(calc_stokes_i_flux_, FP)(\
                in_iquv[0], freq_hz, freq0_hz, lin_spx, num_sp_indices, spxn,\
                have_sp_curvature, sp_curv\
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
            FP rm = rotation_measure_rad[i];\
            if (rm != rm) rm = (FP) 0;\
            const FP wavelength_m = ((FP) C_0) / freq_hz;\
            if (!have_pol_angle || !have_pol_fraction) {\
                /* Compute polarisation angle and fraction if not set. */\
                /* Get reference wavelength for polarisation calculation. */\
                const FP ref_wave_m = (\
                        have_ref_wavelength ? \
                        ref_wavelength_m[i] : (((FP) C_0) / freq0_hz)\
                );\
                const FP freq_at_lambda0_hz = ((FP) C_0) / ref_wave_m;\
                const FP stokes_i_ref = M_CAT(calc_stokes_i_flux_, FP)(\
                        in_iquv[0], freq_hz, freq_at_lambda0_hz, lin_spx,\
                        num_sp_indices, spxn, have_sp_curvature, sp_curv\
                );\
                pol_ang_chi0 = 0.5 * atan2(in_iquv[2], in_iquv[1]) - (\
                        ref_wave_m * ref_wave_m * rm\
                );\
                pol_frac = sqrt(in_iquv[1] * in_iquv[1] + \
                        in_iquv[2] * in_iquv[2]\
                ) / stokes_i_ref;\
            }\
            else {\
                pol_ang_chi0 = pol_angle_rad[i];\
                pol_frac = pol_fraction[i];\
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
