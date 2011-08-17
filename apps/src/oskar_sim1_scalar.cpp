#include "interferometry/oskar_TelescopeModel.h"
#include "station/oskar_StationModel.h"
#include "sky/oskar_SkyModel.h"

#include "apps/oskar_load_telescope.h"
#include "apps/oskar_load_stations.h"

//#include <assert>

int main(int /*argc*/, char** /*argv*/)
{
//    // $> oskar_sim1_scalar settings_file.txt
//
//    // load a settings file with:
//    //  - sky setup
//    //  - telescope layout file
//    //  - stations dir
//    //  - freq
//    //  - ra0  (primary beam phase centre)
//    //  - dec0 (primary beam phase centre)
//
//    const char* telescope_file_path = "";
//    const char* station_dir_path    = "";
//
//
//    // Load telescope layout.
//    oskar_TelescopeModel telescope;
//    load_oskar_telescope(telescope_file_path, &telescope);
//
//    // Load station layouts.
//    oskar_StationModel* stations;
//    int num_stations = load_oskar_stations(station_dir_path, &stations);
//
//    // Check load worked.
//    //ASSERT(num_stations == telescope.num_antennas);
//
//    // Load sky model.
//    oskar_SkyModelGlobal_d sky;
//    // TODO
//
//
//    // convert station positions to ITRS.
//
//
//    // interferometer1_scalar loop.
//
//
//    // Free memory. !TODO!
////    free(telescope.antenna_x);
////    free(telescope.antenna_y);
////    for (unsigned i = 0; i < num_stations; ++i)
////    {
////        free(stations[i].antenna_x);
////        free(stations[i].antenna_y);
////    }

    return 0;
}
