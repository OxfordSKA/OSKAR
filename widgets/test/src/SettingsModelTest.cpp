#include <QtGui/QApplication>
#include <QtCore/QString>
#include <cstdio>
#include <cstdlib>

#include "widgets/oskar_SettingsDelegate.h"
#include "widgets/oskar_SettingsItem.h"
#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsView.h"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Specify OSKAR settings file on command line.\n");
        return EXIT_FAILURE;
    }

    QApplication app(argc, argv);

    oskar_SettingsModel mod;
    mod.registerSetting("global/double_precision", "Use double precision", oskar_SettingsItem::BOOL);
    mod.registerSetting("global/max_sources_per_chunk", "Max. number of sources per chunk", oskar_SettingsItem::INT);
    mod.registerSetting("global/cuda_device_ids", "CUDA device IDs to use", oskar_SettingsItem::INT_CSV_LIST);
    mod.setCaption("global", "Global settings");
    mod.setCaption("sky", "Sky model settings");
    mod.registerSetting("sky/oskar_source_file", "Input OSKAR source file", oskar_SettingsItem::INPUT_FILE_NAME);
    mod.registerSetting("sky/gsm_file", "Input Global Sky Model file", oskar_SettingsItem::INPUT_FILE_NAME);
    mod.registerSetting("sky/output_sky_file", "Output OSKAR source file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    mod.registerSetting("sky/generator", "Generator", oskar_SettingsItem::STRING);
    mod.setCaption("sky/generator/healpix", "HEALPix (all sky) grid");
    mod.registerSetting("sky/generator/healpix/nside", "Nside", oskar_SettingsItem::INT);
    mod.setCaption("sky/generator/random", "Random (broken) power-law in flux");
    mod.registerSetting("sky/generator/random/num_sources", "Number of sources", oskar_SettingsItem::INT);
    mod.registerSetting("sky/generator/random/flux_density_min", "Flux density min (Jy)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/generator/random/flux_density_max", "Flux density max (Jy)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/generator/random/threshold", "Threshold (Jy)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/generator/random/power", "Power law index", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/generator/random/power1", "Power law index 1", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/generator/random/power2", "Power law index 2", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/generator/random/seed", "Random seed", oskar_SettingsItem::RANDOM_SEED);
    mod.setCaption("sky/filter", "Source filter settings");
    mod.registerSetting("sky/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/filter/flux_min", "Flux min (Jy)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("sky/filter/flux_max", "Flux max (Jy)", oskar_SettingsItem::DOUBLE);
    mod.setCaption("telescope", "Telescope model settings");
    mod.registerSetting("telescope/layout_file", "Array layout file", oskar_SettingsItem::INPUT_FILE_NAME);
    mod.registerSetting("telescope/latitude_deg", "Latitude (deg)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("telescope/longitude_deg", "Longitude (deg)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("telescope/altitude_m", "Altitude (m)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("station/station_directory", "Station directory", oskar_SettingsItem::INPUT_DIR_NAME);
    mod.registerSetting("station/disable_station_beam", "Disable station beam", oskar_SettingsItem::BOOL);
    mod.setCaption("station", "Station model settings");
    mod.setCaption("observation", "Observation settings");
    mod.registerSetting("observation/start_frequency", "Start frequency", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("observation/num_channels", "Number of channels", oskar_SettingsItem::INT);
    mod.registerSetting("observation/frequency_inc", "Frequency increment (Hz)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("observation/channel_bandwidth", "Channel bandwidth (Hz)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("observation/phase_centre_ra_deg", "Phase centre RA (deg)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("observation/phase_centre_dec_deg", "Phase centre Dec (deg)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("observation/num_vis_dumps", "Number of visibility dumps", oskar_SettingsItem::INT);
    mod.registerSetting("observation/num_vis_ave", "Number of visibility averages", oskar_SettingsItem::INT);
    mod.registerSetting("observation/num_fringe_ave", "Number of fringe averages", oskar_SettingsItem::INT);
    mod.registerSetting("observation/start_time_utc", "Start time (UTC)", oskar_SettingsItem::DATE_TIME);
    mod.registerSetting("observation/length_seconds", "Observation length (s)", oskar_SettingsItem::DOUBLE);
    mod.registerSetting("observation/oskar_vis_filename", "Output OSKAR visibility file", oskar_SettingsItem::OUTPUT_FILE_NAME);
    mod.registerSetting("observation/ms_filename", "Output Measurement Set name", oskar_SettingsItem::OUTPUT_FILE_NAME);

    oskar_SettingsView view;
    oskar_SettingsDelegate delegate(&view);
    view.setModel(&mod);
    view.setItemDelegate(&delegate);
    view.setWindowTitle("OSKAR Settings");
    view.show();
    view.resizeColumnToContents(0);
    mod.setFile(QString(argv[1]));

    int status = app.exec();
    return status;
}
