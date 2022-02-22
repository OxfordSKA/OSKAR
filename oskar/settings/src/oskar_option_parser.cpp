/*
 * Copyright (c) 2018-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/extern/ezOptionParser/ezOptionParser.hpp"
#include "settings/oskar_option_parser.h"
#include <cstdarg>
#include <vector>
#include <string>

using namespace std;

namespace oskar {

struct OptionParserPrivate : public ez::ezOptionParser
{
    string title, version;
    vector<string> optional_;
    vector<string> optionalHelp_;
    vector<string> required_;
    vector<string> requiredHelp_;
    vector<const char*> input_files_;
    const char* settings_;
    const char* version_;
};

OptionParser::OptionParser(const char* title, const char* ver,
        const char* settings)
{
    p = new OptionParserPrivate;
    p->footer =
            "\n" + string(79, '-') + "\n"
            "OSKAR (version " + ver + ")\n"
            "Copyright (c) 2022, The OSKAR Developers.\n"
            "This program is free and without warranty.\n"
            "" + string(79, '-') + "\n";
    set_version(ver, false);
    set_title(title);
    set_settings(settings);
}

OptionParser::~OptionParser()
{
    delete p;
}

void OptionParser::add_example(const char* text)
{
    p->example += "  " + string(text) + "\n";
}

// Wrapper to define flags with no arguments
void OptionParser::add_flag(const char* flag1, const char* help,
        bool required, const char* flag2)
{
    const char* defaults = "";
    int expectedArgs = 0;
    char delim = 0;
    if (flag2)
    {
        p->add(defaults, required, expectedArgs, delim,
                help, flag1, flag2);
    }
    else
    {
        p->add(defaults, required, expectedArgs, delim, help, flag1);
    }
}

// Wrapper to define flags with arguments with default values.
void OptionParser::add_flag(const char* flag1, const char* help,
        int expected_args, const char* defaults, bool required,
        const char* flag2)
{
    char delim = 0;
    string strHelp = help;
    if (strlen(defaults) > 0 && expected_args == 1 && required == false)
    {
        strHelp += " (default = " + string(defaults) + ")";
    }
    if (flag2)
    {
        p->add(defaults, required, expected_args, delim,
                strHelp.c_str(), flag1, flag2);
    }
    else
    {
        p->add(defaults, required, expected_args, delim,
                strHelp.c_str(), flag1);
    }
}

void OptionParser::add_optional(const char* name, const char* help)
{
    // TODO(BM) Do something with the help field
    p->optional_.push_back(string(name));
    p->optionalHelp_.push_back(string(help));
}

void OptionParser::add_required(const char* name, const char* help)
{
    // TODO(BM) Do something with the help field
    p->required_.push_back(string(name));
    p->requiredHelp_.push_back(string(help));
}

void OptionParser::add_settings_options()
{
    add_required("settings file");
    add_optional("key");
    add_optional("value");
    add_flag("--get", "Print key value in settings file.");
    add_flag("--set", "Set key value in settings file.");
}

bool OptionParser::check_options(int argc, char** argv)
{
    add_flag("--help", "Display usage instructions and exit.", false);
    add_flag("--version", "Display the program name/version banner and exit.",
            false);
    add_flag("--settings", "Display settings and exit.", false);
    p->parse(argc, argv);
    if (is_set("--help"))
    {
        print_usage();
        return false;
    }
    if (is_set("--version"))
    {
        cout << p->version_ << endl;
        return false;
    }
    if (is_set("--settings"))
    {
        cout << string(p->settings_) << endl;
        return false;
    }
    vector<string> bad_opts;
    if (!p->gotRequired(bad_opts))
    {
        for (int i = 0; i < (int)bad_opts.size(); ++i)
        {
            error("Missing required option: %s", bad_opts[i].c_str());
            return false;
        }
    }
    if (!p->gotExpected(bad_opts))
    {
        for (int i = 0; i < (int)bad_opts.size(); ++i)
        {
            error("Got unexpected number of arguments for option: %s",
                    bad_opts[i].c_str());
            return false;
        }
    }
    int min_req_args = (int)p->required_.size();
    if (num_args() < min_req_args)
    {
        error("Expected >= %i input argument(s), %i given", min_req_args,
                num_args());
        return false;
    }
    return true;
}

void OptionParser::error(const char* format, ...) // NOLINT
{
    cerr << "ERROR:\n  ";
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    cerr << "\n\n";
    print_usage();
}

const char* OptionParser::get_arg(int i) const
{
    vector<string*>& first = p->firstArgs;
    vector<string*>& last = p->lastArgs;
    if ((int)first.size() - 1 > i)
    {
        return first[i + 1]->c_str();
    }
    // Requested index is in the last argument set.
    else if (((int)first.size() - 1 + (int)last.size()) > i)
    {
        return last[i - ((int)first.size() - 1)]->c_str();
    }
    return 0;
}

double OptionParser::get_double(const char* name)
{
    double val = 0.0;
    p->get(name)->getDouble(val);
    return val;
}

int OptionParser::get_int(const char* name)
{
    int val = 0;
    p->get(name)->getInt(val);
    return val;
}

const char* OptionParser::get_string(const char* name)
{
    const char* val = 0;
    p->get(name)->getString(val);
    return val;
}

const char* const* OptionParser::get_input_files(int min_required,
        int* num_files)
{
    p->input_files_.clear();
    vector<string*>& first = p->firstArgs;
    vector<string*>& last = p->lastArgs;
    // Note: min_required + 1 because firstArgs[0] is the binary name
    if (((int)first.size() >= min_required + 1) && ((int)last.size() == 0))
    {
        // Note: starts at 1 as index 0 is the binary name.
        for (int i = 1; i < (int)first.size(); ++i)
        {
            p->input_files_.push_back(first[i]->c_str());
        }
    }
    else
    {
        for (int i = 0; i < (int)last.size(); ++i)
        {
            p->input_files_.push_back(last[i]->c_str());
        }
    }
    *num_files = (int) p->input_files_.size();
    return (p->input_files_.size() > 0) ? &(p->input_files_)[0] : 0;
}

int OptionParser::is_set(const char* option)
{
    return p->isSet(option);
}

void OptionParser::print_usage()
{
    string usage;
    p->syntax = p->title + " [OPTIONS]";
    for (int i = 0; i < (int)p->required_.size(); ++i)
    {
        p->syntax += " <" + p->required_[i] + ">";
    }
    for (int i = 0; i < (int)p->optional_.size(); ++i)
    {
        p->syntax += " [" + p->optional_[i] + "]";
    }
    // TODO(BM) overload here rather than editing the library header...!
    p->getUsage(usage);
    cout << usage;
}

int OptionParser::num_args() const
{
    return (int)(p->firstArgs.size() - 1 + p->lastArgs.size());
}

void OptionParser::set_description(const char* description)
{
    p->overview = description;
}

void OptionParser::set_settings(const char* text)
{
    p->settings_ = text;
}

void OptionParser::set_title(const char* text)
{
    p->title = text;
}

void OptionParser::set_version(const char* version, bool show)
{
    if (show)
    {
        p->version = version;
    }
    p->version_ = version;
}

} /* namespace oskar */
