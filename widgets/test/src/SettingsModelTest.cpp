#include <QtGui/QApplication>
#include <QtCore/QString>
#include <cstdio>
#include <cstdlib>

#include "widgets/oskar_SettingsItem.h"
#include "widgets/oskar_SettingsModel.h"
#include "widgets/oskar_SettingsView.h"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    oskar_SettingsModel mod(0);
    mod.registerSetting("global/double_precision", "Use double precision", oskar_SettingsItem::OSKAR_SETTINGS_BOOL);
    mod.registerSetting("global/max_sources_per_chunk", "Max. number of sources per chunk", oskar_SettingsItem::OSKAR_SETTINGS_INT);
    mod.registerSetting("global/cuda_device_ids", "CUDA device IDs to use", oskar_SettingsItem::OSKAR_SETTINGS_INT_CSV_LIST);
    mod.setCaption("global", "Global settings");
    mod.setCaption("sky", "Sky model settings");
    mod.registerSetting("sky/oskar_source_file", "Input OSKAR source file", oskar_SettingsItem::OSKAR_SETTINGS_FILE);
    mod.registerSetting("sky/gsm_file", "Input Global Sky Model file", oskar_SettingsItem::OSKAR_SETTINGS_FILE);
    mod.registerSetting("sky/output_sky_file", "Output OSKAR source file", oskar_SettingsItem::OSKAR_SETTINGS_STRING);
    mod.registerSetting("sky/generator", "Generator", oskar_SettingsItem::OSKAR_SETTINGS_STRING);
    mod.setCaption("sky/generator/healpix", "HEALPix (all sky) grid");
    mod.registerSetting("sky/generator/healpix/nside", "Nside", oskar_SettingsItem::OSKAR_SETTINGS_INT);
    mod.setCaption("sky/generator/random", "Random (broken) power-law in flux");
    mod.registerSetting("sky/generator/random/num_sources", "Number of sources", oskar_SettingsItem::OSKAR_SETTINGS_INT);
    mod.registerSetting("sky/generator/random/flux_density_min", "Flux density min (Jy)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/generator/random/flux_density_max", "Flux density max (Jy)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/generator/random/threshold", "Threshold (Jy)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/generator/random/power", "Power law index", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/generator/random/power1", "Power law index 1", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/generator/random/power2", "Power law index 2", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/generator/random/seed", "Random seed", oskar_SettingsItem::OSKAR_SETTINGS_RANDOM_SEED);
    mod.setCaption("sky/filter", "Source filter settings");
    mod.registerSetting("sky/filter/radius_inner_deg", "Inner radius from phase centre (deg)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/filter/radius_outer_deg", "Outer radius from phase centre (deg)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/filter/flux_min", "Flux min (Jy)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("sky/filter/flux_max", "Flux max (Jy)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.setCaption("telescope", "Telescope model settings");
    mod.registerSetting("telescope/layout_file", "Array layout file", oskar_SettingsItem::OSKAR_SETTINGS_FILE);
    mod.registerSetting("telescope/latitude_deg", "Latitude (deg)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("telescope/longitude_deg", "Longitude (deg)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("telescope/altitude_m", "Altitude (m)", oskar_SettingsItem::OSKAR_SETTINGS_DOUBLE);
    mod.registerSetting("station/station_directory", "Station directory", oskar_SettingsItem::OSKAR_SETTINGS_DIR);
    mod.registerSetting("station/disable_station_beam", "Disable station beam", oskar_SettingsItem::OSKAR_SETTINGS_BOOL);
    mod.setCaption("station", "Station model settings");

    oskar_SettingsView tree_view;
    tree_view.setModel(&mod);
    tree_view.setWindowTitle("OSKAR Settings");
    tree_view.show();
    tree_view.resizeColumnToContents(0);

    int status = app.exec();
    return status;
}
