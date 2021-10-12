/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOADER_STATION_TYPE_MAP_H_
#define OSKAR_TELESCOPE_LOADER_STATION_TYPE_MAP_H_

#include <telescope/oskar_TelescopeLoadAbstract.h>

class TelescopeLoaderStationTypeMap : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderStationTypeMap() {}
    virtual ~TelescopeLoaderStationTypeMap() {}
    virtual void load(oskar_Telescope* telescope, const std::string& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status);
    virtual std::string name() const;
};

#endif /* include guard */
