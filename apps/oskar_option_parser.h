/*
 * Copyright (c) 2012-2017, The University of Oxford
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
 * @file oskar_option_parser.h
 */

#include "extern/ezOptionParser/ezOptionParser.hpp"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdarg>

namespace oskar {

/**
 * @brief
 * Provides a command line parser for OSKAR applications.
 *
 * @details
 * Provides a command line parser for OSKAR applications.
 *
 * Note on the use symbols in option syntax: (the following are advised
 * in order to maintain consistency)
 *
 *  []   = optional (e.g. $ foo [settings file] )
 *  <>   = required (e.g. $ foo <file name> )
 *  ...  = repeating elements, "and so on"
 *  |    = mutually exclusive
 *
 * TODO better handling of unexpected options. It would be useful if a warning
 * could be printed.
 */
class OptionParser : public ez::ezOptionParser
{
public:
    OptionParser(const char* title, const char* ver,
            const char* settings = "")
    {
        this->footer =
                "\n"
                "" + std::string(79, '-') + "\n"
                "OSKAR (version " + ver + ")\n"
                "Copyright (c) 2017, The University of Oxford.\n"
                "This program is free and without warranty.\n"
                "" + std::string(79, '-') + "\n";
        set_version(ver, false);
        set_title(title);
        set_settings(settings);
    }
    virtual ~OptionParser() {}
    void add_example(const char* text)
    {
        this->example += "  " + std::string(text) + "\n";
    }
    // Wrapper to define flags with no arguments
    void add_flag(const char* flag1, const char* help, bool required = false,
            const char* flag2 = 0)
    {
        const char* defaults = "";
        int expectedArgs = 0;
        char delim = 0;
        if (flag2)
            add(defaults, required, expectedArgs, delim, help, flag1, flag2);
        else
            add(defaults, required, expectedArgs, delim, help, flag1);
    }
    // Wrapper to define flags with arguments with default values.
    void add_flag(const char* flag1, const char* help, int expectedArgs,
            const char* defaults = "", bool required = false, const char* flag2 = 0)
    {
        char delim = 0;
        std::string strHelp = help;
        if (strlen(defaults) > 0 && expectedArgs == 1 && required == false)
            strHelp += " (default = " + std::string(defaults) + ")";
        if (flag2)
            add(defaults, required, expectedArgs, delim, strHelp.c_str(), flag1,
                    flag2);
        else
            add(defaults, required, expectedArgs, delim, strHelp.c_str(), flag1);
    }
    void add_optional(const char* name, const char* help = "")
    {
        // TODO Do something with the help field
        optional_.push_back(std::string(name));
        optionalHelp_.push_back(std::string(help));
    }
    void add_required(const char* name, const char* help = "")
    {
        // TODO Do something with the help field
        required_.push_back(std::string(name));
        requiredHelp_.push_back(std::string(help));
    }
    void add_settings_options()
    {
        add_required("settings file");
        add_optional("key");
        add_optional("value");
        add_flag("--get", "Print key value in settings file.");
        add_flag("--set", "Set key value in settings file.");
    }
    bool check_options(int argc, char** argv)
    {
        add_flag("--help", "Display usage instructions and exit.", false);
        add_flag("--version", "Display the program name/version banner and exit.",
                false);
        add_flag("--settings", "Display settings and exit.", false);
        this->parse(argc, argv);
        if (is_set("--help")) {
            print_usage();
            return false;
        }
        if (is_set("--version")) {
            std::cout << version_ << std::endl;
            return false;
        }
        if (is_set("--settings")) {
            std::cout << std::string(settings_) << std::endl;
            return false;
        }
        std::vector<std::string> badOpts;
        if (!gotRequired(badOpts)) {
            for (int i = 0; i < (int)badOpts.size(); ++i) {
                error("Missing required option: %s", badOpts[i].c_str());
                return false;
            }
        }
        if (!gotExpected(badOpts)) {
            for (int i = 0; i < (int)badOpts.size(); ++i)
            {
                error("Got unexpected number of arguments for option: %s",
                        badOpts[i].c_str());
                return false;
            }
        }
        int minReqArgs = (int)required_.size();
        if (num_args() < minReqArgs)
        {
            error("Expected >= %i input argument(s), %i given", minReqArgs,
                    num_args());
            return false;
        }
        return true;
    }
    void error(const char* format, ...)
    {
        std::cerr << "ERROR:\n";
        std::cerr << "  ";
        std::va_list args;
        va_start(args, format);
        std::vprintf(format, args);
        va_end(args);
        std::cerr << "\n\n";
        this->print_usage();
    }
    std::vector<std::string> get_args() const
    {
        std::vector<std::string> args;
        for (int i = 1; i < (int)firstArgs.size(); ++i)
            args.push_back(*this->firstArgs[i]);
        for (int i = 0; i < (int)lastArgs.size(); ++i)
            args.push_back(*this->lastArgs[i]);
        return args;
    }
    const char* get_arg(int i = 0) const
    {
        if ((int)firstArgs.size()-1 > i)
            return (*this->firstArgs[i+1]).c_str();
        // Requested index is in the last argument set.
        else if (((int)firstArgs.size()-1 + (int)lastArgs.size()) > i)
            return (*this->lastArgs[i-((int)firstArgs.size()-1)]).c_str();
        return 0;
    }
    std::vector<std::string> get_input_files(int minRequired = 2) const
    {
        std::vector<std::string> files;
        // Note: minRequired+1 because firstArg[0] == binary name
        bool filesFirst = ((int)this->firstArgs.size() >= minRequired+1) &&
                ((int)this->lastArgs.size() == 0);
        if (filesFirst)
        {
            // Note: starts at 1 as index 0 == the binary name.
            for (int i = 1; i < (int)this->firstArgs.size(); ++i)
                files.push_back(*this->firstArgs[i]);
        }
        else
        {
            for (int i = 0; i < (int)this->lastArgs.size(); ++i)
                files.push_back(*this->lastArgs[i]);
        }
        return files;
    }
    void get_usage(std::string& usage)
    {
        this->syntax = this->title + " [OPTIONS]";
        for (int i = 0; i < (int)required_.size(); ++i)
            this->syntax += " <" + required_[i] + ">";
        for (int i = 0; i < (int)optional_.size(); ++i)
            this->syntax += " [" + optional_[i] + "]";
        // TODO overload here rather than editing the library header...!
        ez::ezOptionParser::getUsage(usage);
    }
    int is_set(const char* option)
    {
        return isSet(option);
    }
    void print_usage()
    {
        std::string usage;
        this->get_usage(usage);
        std::cout << usage;
    }
    int num_args() const
    {
        return (((int)firstArgs.size()-1) + (int)lastArgs.size());
    }
    void set_description(const char* description)
    {
        this->overview = description;
    }
    void set_settings(const char* text)
    {
        this->settings_ = text;
    }
    void set_title(const char* text)
    {
        this->title = text;
    }
    void set_version(const char* version, bool show = true)
    {
        if (show)
            this->version = version;
        version_ = version;
    }

private:
    std::string title, version;
    std::vector<std::string> optional_;
    std::vector<std::string> optionalHelp_;
    std::vector<std::string> required_;
    std::vector<std::string> requiredHelp_;
    const char* settings_;
    const char* version_;
};

} /* namespace oskar */

#endif /* OSKAR_OPTION_PARSER_H_ */
