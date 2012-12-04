/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Demo of short flag name.";
	opt.syntax = "short [OPTIONS] in out";
	opt.add(
		"", // Default.
		0, // Required?
		0, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Simple flag that is set if present. Default is off.", // Help description.
		"-d" // Flag token.
	);

	opt.parse(argc, argv);

	if (opt.lastArgs.size() < 2) {
		std::cerr << "ERROR: Expected 2 arguments, but got " << opt.lastArgs.size() << ".\n\n";
		std::string usage;
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