/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Demo of long name.";
	opt.syntax = "long [OPTIONS] in out";
	opt.example = "long -d a b";
	opt.add(
		"", // Default.
		1, // Required?
		0, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Multi-flag that is set if present. Default is off.", // Help description.
		"-d",     // Flag token.
		"--dbg",  // Flag token.
		"-debug", // Flag token.
		"--debug" // Flag token.
	);

	opt.parse(argc, argv);

	std::string usage;
	if (opt.lastArgs.size() < 2) {
		std::cerr << "ERROR: Expected 2 arguments, but got " << opt.lastArgs.size() << ".\n\n";
		opt.getUsage(usage);
		std::cout << usage;
		return 1;
	} 

	std::vector<std::string> badOptions;
	if(!opt.gotRequired(badOptions)) {
		for(int i=0; i < badOptions.size(); ++i)
			std::cerr << "ERROR: Missing required option " << badOptions[i] << ".\n";
			
		opt.getUsage(usage);
		std::cout << usage;
		return 1;
	}

	if (opt.isSet("-d"))
		std::cout << "-d was set.\n";
		
	std::cout << "First file: " << *opt.lastArgs[0] << std::endl;
	std::cout << "Second file: " << *opt.lastArgs[1] << std::endl;

	return 0;
}