/*
 * Copyright (c) 2017-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_APP_REGISTRAR_H_
#define OSKAR_APP_REGISTRAR_H_

#include <string>
#include <map>

namespace oskar {

struct AppRegistrar
{
    static std::map<std::string, const char*>& apps() {
        static std::map<std::string, const char*> a;
        return a;
    }
    AppRegistrar(const std::string& app, const char* settings) {
        apps().insert(std::pair<std::string, const char*>(app, settings));
    }
};

}

#define M_CAT(A, B) M_CAT_(A, B)
#define M_CAT_(A, B) A##B

#define OSKAR_APP_SETTINGS(app, settings) \
    static oskar::AppRegistrar M_CAT(r_, __LINE__)(#app, settings); // NOLINT

#endif /* include guard */
