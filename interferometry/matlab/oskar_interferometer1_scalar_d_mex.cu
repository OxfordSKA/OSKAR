#include <mex.h>

#include "interferometry/oskar_interferometer1_scalar.h"
#include "interferometry/oskar_horizon_plane_to_itrs.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "station/oskar_StationModel.h"
#include "sky/oskar_SkyModel.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SEC_PER_DAY 86400.0


void getTelescopeModel(const mxArray * matlab_struct, oskar_TelescopeModel_d * model);
void getStationModels(const mxArray * matlab_struct, oskar_StationModel_d * model);
void getGlobalSkyModel(const mxArray * m_struct, oskar_SkyModelGlobal_d * model);

// Entry function.
void mexFunction(int /*num_outputs*/, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    // ==== Parse Inputs.
    if (num_inputs != 12)
    {
        mexPrintf("(telescope, stations, sky, ra0_rad, dec0_red, start_mjd_utc,"
                 "obs length days, num_vis_dumps, num_vis_ave, num_fringe_ave,"
                 "freq, bandwidth)");
        mexErrMsgTxt("Twelve inputs required.");
    }

    // ==== Construct telescope model structure from MATLAB structure.
    oskar_TelescopeModel_d telescope;
    getTelescopeModel(input[0], &telescope);

    // ==== Construct array of station models from MATLAB structure.
    if (mxGetN(input[1]) != telescope.num_antennas)
        mexErrMsgTxt("Dimension mismatch between telescope model and station models");
    size_t mem_size = telescope.num_antennas * sizeof(oskar_StationModel_d);
    oskar_StationModel_d * stations = (oskar_StationModel_d*) malloc(mem_size);
    getStationModels(input[1], stations);

    // ==== Construct global sky model from matlab structure.
    oskar_SkyModelGlobal_d sky;
    getGlobalSkyModel(input[2], &sky);

    double ra0_rad           = mxGetScalar(input[3]);
    double dec0_rad          = mxGetScalar(input[4]);
    double obs_start_mjd_utc = mxGetScalar(input[5]);
    double obs_length_days   = mxGetScalar(input[6]);
    int num_vis_dumps        = (int)mxGetScalar(input[7]);
    int num_vis_ave          = (int)mxGetScalar(input[8]);
    int num_fringe_ave       = (int)mxGetScalar(input[9]);
    double freq              = mxGetScalar(input[10]);
    double bandwidth         = mxGetScalar(input[11]);
    bool disable_e_jones     = false;

    // Setup output arrays.
    const int num_baselines = telescope.num_antennas * (telescope.num_antennas-1) / 2;
    const int rows = 1;
    const int cols = num_baselines * num_vis_dumps;
    //output[0] = mxCreateNumericMatrix(rows, cols, mxDOUBLE_CLASS, mxCOMPLEX);
    output[0] = mxCreateDoubleMatrix(rows, cols, mxCOMPLEX);
    output[1] = mxCreateNumericMatrix(rows, cols, mxDOUBLE_CLASS, mxREAL);
    output[2] = mxCreateNumericMatrix(rows, cols, mxDOUBLE_CLASS, mxREAL);
    output[3] = mxCreateNumericMatrix(rows, cols, mxDOUBLE_CLASS, mxREAL);

    oskar_VisData_d vis;
    double * vis_re = mxGetPr(output[0]);
    double * vis_im = mxGetPi(output[0]);
    vis.u           = mxGetPr(output[1]);
    vis.v           = mxGetPr(output[2]);
    vis.w           = mxGetPr(output[3]);
    vis.amp         = (double2*)malloc(cols * sizeof(double2));

    int err = oskar_interferometer1_scalar_d(telescope, stations, sky, ra0_rad,
            dec0_rad, obs_start_mjd_utc, obs_length_days, num_vis_dumps,
            num_vis_ave, num_fringe_ave, freq, bandwidth, disable_e_jones,
            &vis);

    for (int i = 0; i < cols; ++i)
    {
        vis_re[i] = vis.amp[i].x;
        vis_im[i] = vis.amp[i].y;
    }

    mexPrintf("error %i\n", err);

    // Free memory.
    free(telescope.antenna_x);
    free(telescope.antenna_y);
    free(telescope.antenna_z);
    free(stations);
    free(vis.amp);
}



void getTelescopeModel(const mxArray * matlab_struct, oskar_TelescopeModel_d * model)
{
    // TODO parse structure properly - i.e. check fields etc.
    if (!mxIsStruct(matlab_struct))
        mexErrMsgTxt("Input must be a matlab struct");

    const unsigned num_antennas = (unsigned)mxGetNumberOfElements(mxGetField(matlab_struct, 0, "Xh"));
    const double * xh = mxGetPr(mxGetField(matlab_struct, 0, "Xh"));
    const double * yh = mxGetPr(mxGetField(matlab_struct, 0, "Yh"));
    const double lat  = mxGetScalar(mxGetField(matlab_struct, 0, "latitude"));

    // Allocate memory for telescope model.
    model->num_antennas = num_antennas;
    size_t mem_size = num_antennas * sizeof(double);
    model->antenna_x = (double*)malloc(mem_size);
    model->antenna_y = (double*)malloc(mem_size);
    model->antenna_z = (double*)malloc(mem_size);

    // Convert horizon x, y coordinates to ITRS (local equatorial system)
    oskar_horizon_plane_to_itrs_d(num_antennas, xh, yh, lat, model->antenna_x,
            model->antenna_y, model->antenna_z);
}

void getStationModels(const mxArray * matlab_struct, oskar_StationModel_d * model)
{
    if (!mxIsStruct(matlab_struct))
        mexErrMsgTxt("Input must be a matlab struct");

    const unsigned num_stations = mxGetNumberOfElements(matlab_struct);
    for (unsigned i = 0; i < num_stations; ++i)
    {
        model[i].num_antennas = mxGetNumberOfElements(mxGetField(matlab_struct, i, "X"));
        model[i].antenna_x = mxGetPr(mxGetField(matlab_struct, i, "X"));
        model[i].antenna_y = mxGetPr(mxGetField(matlab_struct, i, "Y"));
    }
}

void getGlobalSkyModel(const mxArray* m_struct, oskar_SkyModelGlobal_d* model)
{
    if (!mxIsStruct(m_struct))
        mexErrMsgTxt("Input must be a matlab struct");

    model->num_sources = (unsigned)mxGetNumberOfElements(mxGetField(m_struct, 0, "RA"));
    mexPrintf("num sources %i\n", model->num_sources);
    model->RA  = mxGetPr(mxGetField(m_struct, 0, "RA"));
    model->Dec = mxGetPr(mxGetField(m_struct, 0, "Dec"));
    model->I   = mxGetPr(mxGetField(m_struct, 0, "I"));
    model->Q   = mxGetPr(mxGetField(m_struct, 0, "Q"));
    model->U   = mxGetPr(mxGetField(m_struct, 0, "U"));
    model->V   = mxGetPr(mxGetField(m_struct, 0, "V"));
}


