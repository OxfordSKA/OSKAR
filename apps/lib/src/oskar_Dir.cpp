/*
 * Copyright (c) 2013, The University of Oxford
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

#include "apps/lib/oskar_Dir.h"

#include <QtCore/QDir>
#include <QtCore/QString>
#include <QtCore/QStringList>

using std::string;
using std::vector;

class oskar_PrivateDir
{
public:
    oskar_PrivateDir(const string& path)
    {
        dir = new QDir(QString::fromStdString(path));
    }
    ~oskar_PrivateDir()
    {
        delete dir;
    }
    QDir* dir;
};

oskar_Dir::oskar_Dir(const string path)
{
    p = new oskar_PrivateDir(path);
}

oskar_Dir::~oskar_Dir()
{
    delete p;
}

string oskar_Dir::absoluteFilePath(const string& filename) const
{
    return p->dir->absoluteFilePath(QString::fromStdString(filename)).
            toStdString();
}

vector<string> oskar_Dir::allSubDirs() const
{
    vector<string> r;
    QStringList dirs = p->dir->entryList(QDir::AllDirs | QDir::NoDotAndDotDot,
            QDir::Name);
    for (int i = 0; i < dirs.size(); ++i)
    {
        r.push_back(dirs[i].toStdString());
    }
    return r;
}

bool oskar_Dir::exists(const string& filename) const
{
    return p->dir->exists(QString::fromStdString(filename));
}

bool oskar_Dir::exists() const
{
    return p->dir->exists();
}

string oskar_Dir::filePath(const string& filename) const
{
    return p->dir->filePath(QString::fromStdString(filename)).toStdString();
}
