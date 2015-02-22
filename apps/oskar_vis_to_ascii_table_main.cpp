/*
 * Copyright (c) 2013-2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include <apps/lib/oskar_OptionParser.h>
#include <oskar_get_error_string.h>
#include <oskar_vis.h>
#include <oskar_binary.h>
#include <oskar_vector_types.h>
#include <oskar_version_string.h>
#include <string>
#include <cstdio>
#include <sstream>
#include <fstream>

template <typename T, typename T2> static T getPolAmp_(T2 amp, int pol_type);
static const char* polStr_(int pol_type);
static void write_header_(FILE* out, int total_vis, int num_chan, int num_times,
        int num_baselines, int num_pol, int num_stations, int num_vis_out,
        int c, double freq_hz, double lambda_m, int p, int t, bool metres);
template <typename T1, typename T2> static void writeData_(int idx, T1 uu,
        T1 vv, T1 ww, T2& a, bool csv, FILE* out);

int main(int argc, char** argv)
{
    // ===== Declare options ==================================================
    oskar_OptionParser opt("oskar_vis_to_ascii_table", oskar_version_string());
    opt.setDescription("Converts an OSKAR visibility binary file to an ASCII "
            "table format with the following columns:\n "
            "[1] index, [2] baseline-uu, [3] baseline-vv, [4] baseline-ww "
            "[5] Real, [6] Imag. "
            "The table is written out in baseline-time order where baseline "
            "is the fastest varying dimension");
    opt.addRequired("OSKAR vis file");
    opt.addOptional("output file name");
    opt.addFlag("-c", "Channel index to write to file. (default = 0)", 1, "0",
            false, "--channel");
    opt.addFlag("-p", "Polarisation ID to write to file. (default = 0) "
            "(0=I, 1=Q, 2=U, 3=V, 4=XX, 5=XY, 6=YX, 7=YY)",
            1, "0", false, "--polarisation");
    opt.addFlag("-t", "Time index to write to file. (default = all)", 1, "",
            false, "--time");
    opt.addFlag("-w", "Output baseline coordinates in wavelengths. "
            "(default = metres)", false, "--baseline_wavelengths");
    opt.addFlag("-h", "Write a summary header in the ASCII table. ");
    opt.addFlag("-v", "Verbose mode.");
    opt.addFlag("--csv", "Write in CSV format");
    opt.addFlag("-s", "Write output table to standard output instead of to file.",
            false, "--stdout");

    if (!opt.check_options(argc, argv))
        return OSKAR_FAIL;

    // ===== Read options ====================================================
    const char* vis_file = opt.getArg(0);
    std::string txt_file;
    if (opt.numArgs() == 2)
        txt_file = std::string(opt.getArg(1));
    else {
        txt_file = std::string(vis_file) + ".txt";
    }
    int c = 0;
    if (opt.isSet("-c"))
        opt.get("-c")->getInt(c);
    int p = 0;
    if (opt.isSet("-p"))
        opt.get("-p")->getInt(p);
    int t = -1;
    if (opt.isSet("-t"))
        opt.get("-t")->getInt(t);
    bool metres = !opt.isSet("-w");
    bool write_header = opt.isSet("-h");
    bool csv = opt.isSet("--csv");
    bool verbose = opt.isSet("-v");

    // ===== Write table ======================================================
    int status = 0;
    oskar_Binary* h = oskar_binary_create(vis_file, 'r', &status);
    oskar_Vis* vis = oskar_vis_read(h, &status);
    if (status)
    {
        fprintf(stderr, "ERROR: Unable to read specified visibility file: %s\n",
                vis_file);
        oskar_vis_free(vis, &status);
        oskar_binary_free(h);
        return status;
    }
    oskar_binary_free(h);

    int num_chan = oskar_vis_num_channels(vis);
    int num_times = oskar_vis_num_times(vis);
    int num_baselines = oskar_vis_num_baselines(vis);
    int num_pol = oskar_vis_num_pols(vis);
    int num_stations = oskar_vis_num_stations(vis);
    int total_vis = num_chan * num_times * num_baselines * num_pol;
    double freq_start_hz = oskar_vis_freq_start_hz(vis);
    double freq_inc_hz = oskar_vis_freq_inc_hz(vis);
    double freq_hz = freq_start_hz + c * freq_inc_hz;
    double lambda_m = 299792458.0 / freq_hz;

    if (t != -1 && t > num_times-1) {
        fprintf(stderr, "ERROR: Time index out of range.\n");
        return OSKAR_FAIL;
    }
    if (c > num_chan-1) {
        fprintf(stderr, "ERROR: Channel index out of range.\n");
        return OSKAR_FAIL;
    }


    FILE* out;
    if (!opt.isSet("-s")) {
        out = fopen(txt_file.c_str(), "w");
        if (out == NULL) return OSKAR_FAIL;
    }
    else {
        out = stdout;
    }

    const oskar_Mem* uu = oskar_vis_baseline_uu_metres_const(vis);
    const oskar_Mem* vv = oskar_vis_baseline_vv_metres_const(vis);
    const oskar_Mem* ww = oskar_vis_baseline_ww_metres_const(vis);
    const oskar_Mem* amp = oskar_vis_amplitude_const(vis);
    // amplitudes dims: channel x times x baselines x pol
    int amp_offset = c * num_times * num_baselines;
    if (t != -1) amp_offset += t * num_baselines;
    // baseline dims: times x baselines
    int baseline_offset = 0;
    if (t != -1) baseline_offset = t * num_baselines;
    int type = oskar_mem_type(uu);

    int num_vis_out = num_baselines;
    if (t == -1) num_vis_out *= num_times;

    if (verbose) {
        write_header_(stdout, total_vis, num_chan, num_times, num_baselines,
                num_pol, num_stations, num_vis_out, c, freq_hz, lambda_m, p, t,
                metres);
#if 0
        fprintf(stdout, "amp_offset      = %i\n", amp_offset);
        fprintf(stdout, "baseline_offset = %i\n", baseline_offset);
#endif
    }

    // Write header if specified
    if (write_header)
    {
        write_header_(out, total_vis, num_chan, num_times, num_baselines,
                num_pol, num_stations, num_vis_out, c, freq_hz, lambda_m, p, t,
                metres);
        char pre = '#';
        fprintf(out, "%c\n", pre);
        fprintf(out, "%c %s %-14s %-15s %-15s %-23s %-15s\n",
                pre, "Idx", " uu", "  vv", "  ww", "  Amp. Re.", "  Amp. Im.");
    }

    if (type == OSKAR_DOUBLE)
    {
        const double* uu_ = oskar_mem_double_const(uu, &status);
        const double* vv_ = oskar_mem_double_const(vv, &status);
        const double* ww_ = oskar_mem_double_const(ww, &status);
        const double4c* amp_ = oskar_mem_double4c_const(amp, &status);
        int aIdx = amp_offset;
        int bIdx = baseline_offset;
        for (int i = 0; i < num_vis_out; ++i, ++bIdx, ++aIdx)
        {
            double2 a = getPolAmp_<double2, double4c>(amp_[aIdx], p);
            double buu = (metres)? uu_[bIdx] : uu_[bIdx]/lambda_m;
            double bvv = (metres)? vv_[bIdx] : vv_[bIdx]/lambda_m;
            double bww = (metres)? ww_[bIdx] : ww_[bIdx]/lambda_m;
            writeData_<double, double2>(i, buu, bvv, bww, a, csv, out);
        }
    }
    else // OSKAR_SINGLE
    {
        const float* uu_ = oskar_mem_float_const(uu, &status);
        const float* vv_ = oskar_mem_float_const(vv, &status);
        const float* ww_ = oskar_mem_float_const(ww, &status);
        const float4c* amp_ = oskar_mem_float4c_const(amp, &status);
        int aIdx = amp_offset;
        int bIdx = baseline_offset;
        for (int i = 0; i < num_vis_out; ++i, ++bIdx, ++aIdx)
        {
            float2 a = getPolAmp_<float2, float4c>(amp_[aIdx], p);
            float buu = (metres)? uu_[bIdx] : uu_[bIdx]/lambda_m;
            float bvv = (metres)? vv_[bIdx] : vv_[bIdx]/lambda_m;
            float bww = (metres)? ww_[bIdx] : ww_[bIdx]/lambda_m;
            writeData_<float, float2>(i, buu, bvv, bww, a, csv, out);
        }
    }

    fclose(out);
    oskar_vis_free(vis, &status);

    return status;
}

template <typename T1, typename T2> static void writeData_(int idx, T1 uu,
        T1 vv, T1 ww, T2& a, bool csv, FILE* out)
{
    // index, u, v, Re, Im, weight
    if (csv)
    {
        char sep = ',';
        fprintf(out, "%i", idx);
        fprintf(out, "%c%.8e", sep, uu);
        fprintf(out, "%c%.8e", sep, vv);
        fprintf(out, "%c%.8e", sep, ww);
        fprintf(out, "%c%.16e", sep, a.x);
        fprintf(out, "%c%.16e", sep, a.y);
        fprintf(out, "\n");
    }
    else
    {
        char sep = ' ';
        fprintf(out, "%-5i", idx);
        fprintf(out, "%c% -.8e", sep, uu);
        fprintf(out, "%c% -.8e", sep, vv);
        fprintf(out, "%c% -.8e", sep, ww);
        fprintf(out, "%c% -.16e", sep, a.x);
        fprintf(out, "%c% -.16e", sep, a.y);
        fprintf(out, "\n");
    }
}

template <typename T, typename T2> static T getPolAmp_(T2 amp, int pol_type)
{
    T a;
    switch (pol_type)
    {
    case 0: // stokes-I = 0.5 (XX + YY)
    {
        a.x = 0.5 * (amp.a.x + amp.d.x);
        a.y = 0.5 * (amp.a.y + amp.d.y);
        break;
    }
    case 1: // stokes-Q = 0.5 (YY - YY)
    {
        a.x = 0.5 * (amp.a.x - amp.d.x);
        a.y = 0.5 * (amp.a.y - amp.d.y);
        break;
    }
    case 2: // stokes-U = 0.5 (XY + YX)
    {
        a.x = 0.5 * (amp.b.x + amp.c.x);
        a.y = 0.5 * (amp.b.y + amp.c.y);
        break;
    }
    case 3: // stokes-V = 0.5i (XY - YX)
    {
        a.x = 0.5 * (amp.b.y - amp.c.y);
        a.x = -0.5 * (amp.b.x - amp.c.x);
        break;
    }
    case 4: // XX
    {
        a.x = amp.a.x;
        a.y = amp.a.y;
        break;
    }
    case 5: // XY
    {
        a.x = amp.b.x;
        a.y = amp.b.y;
        break;
    }
    case 6: // YX
    {
        a.x = amp.c.x;
        a.y = amp.c.y;
        break;
    }
    case 7: // YY
    {
        a.x = amp.d.x;
        a.y = amp.d.y;
        break;
    }
    default:
    {
        a.x = 0.0;
        a.y = 0.0;
        break;
    }
    };
    return a;
}

static const char* polStr_(int pol_type)
{
    switch (pol_type)
    {
    case 0:  return "Stokes-I";
    case 1:  return "Stokes-Q";
    case 2:  return "Stokes-U";
    case 3:  return "Stokes-V";
    case 4:  return "XX";
    case 5:  return "XY";
    case 6:  return "YX";
    case 7:  return "YY";
    default: return "Undef";
    };
    return "Undef";
}

static void write_header_(FILE* out, int total_vis, int num_chan, int num_times,
        int num_baselines, int num_pol, int num_stations, int num_vis_out,
        int c, double freq_hz, double lambda_m, int p, int t, bool metres)
{
    char pre = '#';
    fprintf(out, "%c %s:\n", pre, "Totals");
    fprintf(out, "%c   %-20.20s %i (%i)\n", pre, "No. vis. amplitudes",
            total_vis, total_vis/4);
    fprintf(out, "%c   %-20.20s %i\n", pre, "No. channels", num_chan);
    fprintf(out, "%c   %-20.20s %i\n", pre, "No. times", num_times);
    fprintf(out, "%c   %-20.20s %i\n", pre, "No. baselines", num_baselines);
    fprintf(out, "%c   %-20.20s %i\n", pre, "No. pols.", num_pol);
    fprintf(out, "%c   %-20.20s %i\n", pre, "No. stations", num_stations);
    fprintf(out, "%c %s:\n", pre, "Written");
    fprintf(out, "%c   %-20.20s %i\n", pre, "No. vis", num_vis_out);
    fprintf(out, "%c   %-20.20s %i\n", pre, "Channel", c);
    fprintf(out, "%c   %-20.20s %f\n", pre, "Frequency (MHz)", freq_hz/1.0e6);
    fprintf(out, "%c   %-20.20s %f\n", pre, "Wavelength (m)", lambda_m);
    std::ostringstream ss;
    if (t == -1) ss << "all" << " (" << 0 << ".." << num_times-1 << ")";
    else ss << t;
    fprintf(out, "%c   %-20.20s %s\n", pre, "Time", ss.str().c_str());
    fprintf(out, "%c   %-20.20s %s\n", pre, "Polarisation", polStr_(p));
    fprintf(out, "%c   %-20.20s %s\n", pre, "Baseline units", metres ?
            ("metres"):("wavelengths"));
}
