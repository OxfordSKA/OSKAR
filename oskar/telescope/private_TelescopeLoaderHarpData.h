/*
 * Copyright (c) 2022-2024, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOADER_HARP_DATA_H_
#define OSKAR_TELESCOPE_LOADER_HARP_DATA_H_

#include <vector>

#include "oskar/harp/oskar_harp.h"
#include "oskar/telescope/oskar_TelescopeLoadAbstract.h"
#include "oskar/utility/oskar_thread.h"

class TelescopeLoaderHarpData : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderHarpData();
    virtual ~TelescopeLoaderHarpData();
    virtual void load(oskar_Telescope* telescope,
            const std::string& cwd, int num_subdirs,
            std::map<std::string, std::string>& filemap, int* status);
    virtual void load(oskar_Station* station, const std::string& cwd,
            int num_subdirs, int depth,
            std::map<std::string, std::string>& filemap, int* status);
    virtual std::string name() const;

private:
    oskar_Harp* get_harp_model(const std::string& path, int precision);
    std::vector<std::string> get_path_list(
            const std::map<std::string, std::string>& filemap, int* status);
    void update_map(std::map<std::string, std::string>& files,
            const std::string& cwd);

private:
    oskar_Mutex* mutex;
    std::string wildcard;
    std::map<std::string, oskar_Harp*> model_map;
};

#endif /* include guard */
