/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOADER_ELEMENT_PATTERN_H_
#define OSKAR_TELESCOPE_LOADER_ELEMENT_PATTERN_H_

#include <telescope/oskar_TelescopeLoadAbstract.h>

#include <vector>

class TelescopeLoaderElementPattern : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderElementPattern();
    virtual ~TelescopeLoaderElementPattern();
    virtual void load(oskar_Telescope* telescope, const std::string& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status);
    virtual void load(oskar_Station* station, const std::string& cwd,
            int num_subdirs, int depth,
            std::map<std::string, std::string>& filemap, int* status);
    virtual std::string name() const;

private:
    void load_element_patterns(oskar_Station* station,
            const std::map<std::string, std::string>& filemap, int* status);
    void load_fitted_data(int feed, oskar_Station* station,
            const std::vector<std::string>& keys,
            const std::vector<std::string>& paths, int* status);
    void load_functional_data(int feed, oskar_Station* station,
            const std::vector<std::string>& keys,
            const std::vector<std::string>& paths, int* status);
    void load_spherical_wave_data(oskar_Station* station,
            const std::vector<std::string>& keys,
            const std::vector<std::string>& paths, int* status);
    static void parse_filename(const char* s, char** buffer, size_t* buflen,
            int* index, double* freq, int* status);
    void update_map(std::map<std::string, std::string>& files,
            const std::string& cwd);

private:
    std::string wildcard;
    std::string fit_root_x, fit_root_y, fit_root_scalar;
    std::string root, root_x, root_y;
    oskar_Telescope* telescope_;
};

#endif /* OSKAR_TELESCOPE_LOADER_ELEMENT_PATTERN_H_ */
