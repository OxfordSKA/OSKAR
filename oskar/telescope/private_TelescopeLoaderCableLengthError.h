/*
 * Copyright (c) 2019-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOADER_CABLE_LENGTH_ERROR_H_
#define OSKAR_TELESCOPE_LOADER_CABLE_LENGTH_ERROR_H_

#include <telescope/oskar_TelescopeLoadAbstract.h>

class TelescopeLoaderCableLengthError : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderCableLengthError() {}
    virtual ~TelescopeLoaderCableLengthError() {}
    virtual void load(oskar_Telescope* telescope, const std::string& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status
    );
    virtual void load(oskar_Station* station, const std::string& cwd,
            int num_subdirs, int depth,
            std::map<std::string, std::string>& filemap, int* status);
    virtual std::string name() const;
};

#endif /* include guard */
