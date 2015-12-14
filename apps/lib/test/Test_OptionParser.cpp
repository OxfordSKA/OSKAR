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


#include <apps/lib/oskar_OptionParser.h>

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std;

// Test case selection.
#define TEST_CASE_3

int main(int argc, char** argv)
{
    oskar_OptionParser opt("Test_OptionParser");


    // ========================================================================
    //
    // Improvement ideas:
    //  * Add some specification of number of required and optional arguments.
    //  * Allow optional arguments to be before required.
    //
    // ========================================================================


#if defined(TEST_CASE_1)
    {
        // Description
        //   Two required options
        // Expect:
        //   Should be OK but no default checking between setting options
        //   and checking options.
        //   Will parse the check if at least 2 arguments are specified.
        //   The meaning put on these is undefined.

        opt.addRequired("required 1");
        opt.addRequired("required 2");

        if (!opt.check_options(argc, argv))
            return EXIT_FAILURE;

        vector<string> r = opt.getArgs();
        cout << "Number of args = " << r.size() << endl;
        for (int i = 0; i < (int)r.size(); ++i) {
            cout << "---> r[" << i << "] = "  << r[i] << endl;
        }
    }
#elif defined(TEST_CASE_2)
    // Description
    //   One required and one optional argument.
    // Expect:
    //   Pass the check if at least one argument is given.
    // Note:
    //   Required arguments are always printed first in the list.
    //   This prevents some combination of arguments. e.g. a list of
    //   required arguments in combination with an optional argument.

    opt.addRequired("required");
    opt.addOptional("optional");

    if (!opt.check_options(argc, argv))
        return EXIT_FAILURE;

    vector<string> args = opt.getArgs();
    cout << "Number of args = " << args.size() << endl;
    for (int i = 0; i < (int)args.size(); ++i) {
        cout << "---> arg[" << i << "] = "  << args[i] << endl;
    }

#elif defined(TEST_CASE_3)
    {
        // Description
        //   Required option and required flag option.
        // Expect:
        //   This should work in all cases as flags and required values or lists
        //   are easily distinguished.

        opt.addRequired("required 1");
        opt.addFlag("-f", "required flag", 1, "", true);

        if (!opt.check_options(argc, argv))
            return EXIT_FAILURE;

        vector<string> args = opt.getArgs();
        cout << "Number of args = " << args.size() << endl;
        for (int i = 0; i < (int)args.size(); ++i) {
            cout << "---> arg[" << i << "] = "  << args[i] << endl;
        }

        string flag;
        opt.get("-f")->getString(flag);
        cout << "Flag -f: " << flag << endl;
    }
#endif

    return EXIT_SUCCESS;
}
