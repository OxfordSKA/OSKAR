/*
 * Copyright (c) 2012, The University of Oxford
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


#ifndef OSKAR_OPTION_PARSER_H_
#define OSKAR_OPTION_PARSER_H_

/**
 * @file oskar_OptionParser.h
 */

#include "oskar_global.h"
#include "extern/ezOptionParser-0.2.0/ezOptionParser.hpp"
#include <vector>
#include <string>

class OSKAR_EXPORT oskar_OptionParser : public ez::ezOptionParser
{
public:
    oskar_OptionParser()
    {
        this->footer =
                "\n"
                "|" + std::string(80, '-') + "\n"
                "| OSKAR (version " + OSKAR_VERSION_STR + ")\n"
                "| Copyright (C) 2012, The University of Oxford.\n"
                "| This program is free and without warranty.\n"
                "|" + std::string(80, '-') + "\n";

        this->add("", 0, 0, 0, "Display usage instructions.", "-h", "-help", "--help",
                "--usage");
    }
    virtual ~oskar_OptionParser() {}
    void setOverview(const char* text)
    {
        std::string line(80, '-');
        this->overview = line + "\n" + text + "\n" + line;
    }
    void setSyntax(const char* text)
    {
        this->syntax = "\n  $ " + std::string(text);
    }
    void addExample(const char* text)
    {
        this->example += "  $ " + std::string(text) + "\n";
    }
    void printUsage()
    {
        std::string usage;
        this->getUsage(usage);
        std::cout << usage;
    }
    std::vector<std::string> getInputFiles(int minRequired = 2) const
    {
        std::vector<std::string> files;
        // Note: minRequired+1 because firstArg[0] == binary name
        bool filesFirst = ((int)this->firstArgs.size() >= minRequired+1) &&
                ((int)this->lastArgs.size() == 0);
        if (filesFirst)
        {
            // Note: starts at 1 as index 0 == the binary name.
            for (int i = 1; i < (int)this->firstArgs.size(); ++i)
            {
                files.push_back(*this->firstArgs[i]);
            }
        }
        else
        {
            for (int i = 0; i < (int)this->lastArgs.size(); ++i)
            {
                files.push_back(*this->lastArgs[i]);
            }
        }
        return files;
    }
};

#endif /* OSKAR_OPTION_PARSER_H_ */
