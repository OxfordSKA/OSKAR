/*
 * Copyright (c) 2013-2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "apps/lib/oskar_dir.h"

#include <QtCore/QDir>
#include <QtCore/QString>
#include <QtCore/QStringList>

#include <cstdio>

using std::string;
using std::vector;

extern "C"
int oskar_dir_remove(const char* dir_name)
{
    return (int) oskar_Dir::rmtree(std::string(dir_name));
}

extern "C"
int oskar_dir_exists(const char* dir_name)
{
    oskar_Dir dir(dir_name);
    return dir.exists();
}

struct oskar_Dir::oskar_DirPrivate
{
    oskar_DirPrivate(const string& path) : dir(QString::fromStdString(path)) {}
    QDir dir;
};

oskar_Dir::oskar_Dir(const string& path)
{
    p = new oskar_DirPrivate(path);
}

oskar_Dir::~oskar_Dir()
{
    delete p;
}

string oskar_Dir::absoluteFilePath(const string& filename) const
{
    return p->dir.absoluteFilePath(QString::fromStdString(filename)).
            toStdString();
}

string oskar_Dir::absolutePath() const
{
    return p->dir.absolutePath().toStdString();
}

vector<string> oskar_Dir::allFiles() const
{
    vector<string> r;
    QStringList files = p->dir.entryList(QDir::Files | QDir::NoDotAndDotDot |
            QDir::System | QDir::Hidden, QDir::Name);
    for (int i = 0; i < files.size(); ++i)
    {
        r.push_back(files[i].toStdString());
    }
    return r;
}

vector<string> oskar_Dir::allSubDirs() const
{
    vector<string> r;
    QStringList dirs = p->dir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot,
            QDir::Name);
    for (int i = 0; i < dirs.size(); ++i)
    {
        r.push_back(dirs[i].toStdString());
    }
    return r;
}

bool oskar_Dir::exists(const string& filename) const
{
    return p->dir.exists(QString::fromStdString(filename));
}

bool oskar_Dir::exists() const
{
    return p->dir.exists();
}

string oskar_Dir::filePath(const string& filename) const
{
    return p->dir.filePath(QString::fromStdString(filename)).toStdString();
}

vector<string> oskar_Dir::filesStartingWith(const string& rootname) const
{
    vector<string> r;
    QStringList files = p->dir.entryList(
            QStringList() << (QString::fromStdString(rootname) + "*"),
            QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
    for (int i = 0; i < files.size(); ++i)
    {
        r.push_back(files[i].toStdString());
    }
    return r;
}

bool oskar_Dir::rmdir(const string& name)
{
    return p->dir.rmdir(QString::fromStdString(name));
}

bool oskar_Dir::rmtree(const string& root)
{
    bool result = 0;
    oskar_Dir dir(root);
    if (dir.exists())
    {
        // Remove all files.
        vector<string> files = dir.allFiles();
        for (size_t i = 0; i < files.size(); ++i)
        {
            const string& name = files[i];
            if (std::remove(dir.absoluteFilePath(name).c_str())) return false;
        }

        // Recursively remove all subdirectories.
        vector<string> dirs = dir.allSubDirs();
        for (size_t i = 0; i < dirs.size(); ++i)
        {
            const string& name = dirs[i];
            if (!rmtree(dir.absoluteFilePath(name))) return false;
        }

        // Remove the empty directory.
        result = dir.rmdir(dir.absolutePath());
    }
    return result;
}
