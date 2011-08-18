
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_load_sources.h"

#include "station/oskar_StationModel.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_interferometer1_scalar.h"

#include "apps/oskar_load_telescope.h"
#include "apps/oskar_load_stations.h"

#include <cstdio>
#include <cstdlib>

#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QSettings>
#include <QtCore/QTime>


int main(int argc, char** argv)
{
    const double deg2rad = 0.0174532925199432957692;

    // $> oskar_sim1_scalar settings_file.txt
    if (argc != 2)
    {
        fprintf(stderr, "ERROR: missing command line arguments\n");
        fprintf(stderr, "Usage:  $ sim1_scalar [settings file]\n");
        return EXIT_FAILURE;
    }

    QString settings_file = QString(argv[1]);
    if (!QFileInfo(settings_file).isFile())
    {
        fprintf(stderr, "ERROR: specified settings file doesn't exist!\n");
        return EXIT_FAILURE;
    }

    QSettings settings(settings_file, QSettings::IniFormat);
    QString sky_file       = settings.value("sky/source_file").toString();

    QString telescope_file  = settings.value("telescope/layout_file").toString();
    double latitude         = settings.value("telescope/latitude").toDouble();
    double longitude        = settings.value("telescope/longitude").toDouble();

    QString station_dir     = settings.value("station/station_directory").toString();

    double frequency        = settings.value("observation/frequency").toDouble();
    double bandwidth        = settings.value("observation/channel_bandwidth").toDouble();
    double ra0              = settings.value("observation/phase_centre_ra").toDouble();
    double dec0             = settings.value("observation/phase_centre_dec").toDouble();
    double obs_length_sec   = settings.value("observation/length_seconds").toDouble();
    double obs_start_mjd    = settings.value("observation/start_mjd_utc").toDouble();
    QString output_file     = settings.value("observation/output_file").toString();

    unsigned num_vis_dumps  = settings.value("correlator/num_vis_dumps").toUInt();
    unsigned num_vis_ave    = settings.value("correlator/num_vis_ave").toUInt();
    unsigned num_fringe_ave = settings.value("correlator/num_fringe_ave").toUInt();


    printf("\n");
    printf("= settings (%s)\n", settings_file.toLatin1().data());
    printf("  - Sky file           = %s\n", sky_file.toLatin1().data());
    printf("  - Stations directory = %s\n", station_dir.toLatin1().data());
    printf("  - Telescope file     = %s\n", telescope_file.toLatin1().data());
    printf("  - Frequency (Hz)     = %e\n", frequency);
    printf("  - Bandwidth (Hz)     = %f\n", bandwidth);
    printf("  - Ra0 (deg)          = %f\n", ra0);
    printf("  - Dec0 (deg)         = %f\n", dec0);
    printf("  - Latitude (deg)     = %f\n", latitude);
    printf("  - Longitude (deg)    = %f\n", longitude);
    printf("  - Obs. length (sec)  = %f\n", obs_length_sec);
    printf("  - Obs. start (mjd)   = %f\n", obs_start_mjd);
    printf("  - num_vis_dumps      = %i\n", num_vis_dumps);
    printf("  - num_vis_ave        = %i\n", num_vis_ave);
    printf("  - num_fringe_ave     = %i\n", num_fringe_ave);
    printf("\n");

    // Load sky model.
    oskar_SkyModelGlobal_d sky;
    if (!QFileInfo(sky_file).isFile())
    {
        fprintf(stderr, "ERROR: sky file doesn't exist!\n");
        return EXIT_FAILURE;
    }
    oskar_load_sources(sky_file.toLatin1().data(), &sky);

    printf("= Number of sources in model = %i\n", sky.num_sources);


    // Load telescope layout.
    oskar_TelescopeModel telescope;
    telescope.latitude  = latitude;
    telescope.longitude = longitude;
    if (!QFileInfo(telescope_file).isFile())
    {
        fprintf(stderr, "ERROR: telescope layout file doesn't exist!\n");
        return EXIT_FAILURE;
    }
    oskar_load_telescope(telescope_file.toLatin1().data(), &telescope);
    // FIXME convert stations to wavenumbers ???

    // Load station layouts.
    oskar_StationModel* stations;
    if (!QFileInfo(station_dir).isDir())
    {
        fprintf(stderr, "ERROR: station directory doesn't exist!\n");
        return EXIT_FAILURE;
    }
    unsigned num_stations = oskar_load_stations(station_dir.toLatin1().data(), &stations);
    // FIXME convert stations to wavenumbers.


    // Check load worked.
    if (num_stations != telescope.num_antennas)
    {
        fprintf(stderr, "ERROR: Error loading telescope geometry\n");
        return EXIT_FAILURE;
    }

    const unsigned num_baselines = num_stations * (num_stations - 1) / 2;
    const unsigned num_vis_coordinates = num_baselines * num_vis_dumps;
    double2* vis = (double2*) malloc(num_vis_coordinates * sizeof(double2));
    double* u    = (double*)  malloc(num_vis_coordinates * sizeof(double));
    double* v    = (double*)  malloc(num_vis_coordinates * sizeof(double));
    double* w    = (double*)  malloc(num_vis_coordinates * sizeof(double));

    double obs_length_days = obs_length_sec / (24.0 * 60.0 * 60.0);

    QTime timer;
    timer.start();
    int err = oskar_interferometer1_scalar_d(telescope, stations,
            sky, ra0 * deg2rad, dec0 * deg2rad, obs_start_mjd, obs_length_days,
            num_vis_dumps, num_vis_ave, num_fringe_ave, frequency,
            bandwidth, vis, u, v, w);

    printf("= Completed simulation after %f seconds. [error code = %i]\n",
            timer.elapsed() / 1.0e3, err);

    printf("= number of visibility points generated = %i\n", num_vis_coordinates);

//    for (unsigned i = 0; i < num_vis_coordinates; ++i)
//    {
//        printf("% -10.3f % -10.3f % -10.3f % -10.3f % -10.3f\n",
//                u[i], v[i], w[i], vis[i].x, vis[i].y);
//    }


    // Write memory out.
    FILE * file;
    file = fopen(output_file.toLatin1().data(), "wb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open output file.\n");
        return EXIT_FAILURE;
    }

    for (unsigned i = 0; i < num_vis_coordinates; ++i)
    {
        fwrite(&u[i],     sizeof(double), 1, file);
        fwrite(&v[i],     sizeof(double), 1, file);
        fwrite(&w[i],     sizeof(double), 1, file);
        fwrite(&vis[i].x, sizeof(double), 1, file);
        fwrite(&vis[i].y, sizeof(double), 1, file);
    }

//    size_t mem_size = num_vis_coordinates * sizeof(double);
//    fwrite((const void*)u, sizeof(double), mem_size, file);
//    fwrite((const void*)v, sizeof(double), mem_size, file);
//    fwrite((const void*)w, sizeof(double), mem_size, file);
//    mem_size = num_vis_coordinates * sizeof(double2);
//    fwrite((const void*)vis, sizeof(double2), mem_size, file);


    fclose(file);

    // Free memory.
    free(vis);
    free(u);
    free(v);
    free(w);

    free(sky.RA);
    free(sky.Dec);
    free(sky.I);
    free(sky.Q);
    free(sky.U);
    free(sky.V);

    free(telescope.antenna_x);
    free(telescope.antenna_y);
    free(telescope.antenna_z);

    for (unsigned i = 0; i < num_stations; ++i)
    {
        free(stations[i].antenna_x);
        free(stations[i].antenna_y);
    }

    return EXIT_SUCCESS;
}
