/*
 * Copyright (c) 2020-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOADER_GAIN_MODEL_H_
#define OSKAR_TELESCOPE_LOADER_GAIN_MODEL_H_

#include <telescope/oskar_TelescopeLoadAbstract.h>

class TelescopeLoaderGainModel : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderGainModel() {}
    virtual ~TelescopeLoaderGainModel() {}
    virtual void load(oskar_Telescope* telescope, const std::string& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status);
    virtual void load(oskar_Station* station, const std::string& cwd,
            int num_subdirs, int depth,
            std::map<std::string, std::string>& filemap, int* status);
    virtual std::string name() const;
};

#endif /* include guard */
