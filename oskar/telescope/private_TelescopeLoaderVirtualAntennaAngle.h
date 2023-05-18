/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOADER_VIRTUAL_ANTENNA_ANGLE_H_
#define OSKAR_TELESCOPE_LOADER_VIRTUAL_ANTENNA_ANGLE_H_

#include <telescope/oskar_TelescopeLoadAbstract.h>

class TelescopeLoaderVirtualAntennaAngle : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderVirtualAntennaAngle() {}
    virtual ~TelescopeLoaderVirtualAntennaAngle() {}
    virtual void load(oskar_Telescope* telescope, const std::string& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status);
    virtual std::string name() const;
};

#endif /* include guard */
