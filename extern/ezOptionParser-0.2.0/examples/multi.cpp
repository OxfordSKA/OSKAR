/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Demo of multiple instances of flag.";
	opt.syntax = "multi [OPTIONS]";
	opt.example = "multi -p 0,0,0 -p 2,3,4 -pt 1,1,1 --pnt 1,2,3 --point -1,1e6,-.314\n";

	opt.add(
		"0,0,0", // Default.
		1, // Required?
		3, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Point coordinates (3-tuple). For example: -p 1,2,3.", // Help description.
		"-p",     // Flag token.
		"-pt",    // Flag token.
		"--pnt",  // Flag token.
		"--point" // Flag token.
	);

	opt.parse(argc, argv);

	std::string usage;

	std::vector<std::string> badOptions;
	if(!opt.gotRequired(badOptions)) {
		for(int i=0; i < badOptions.size(); ++i)
			std::cerr << "ERROR: Missing required option " << badOptions[i] << ".\n\n";
			
		opt.getUsage(usage);
		std::cout << usage;
		return 1;
	}

	if(!opt.gotExpected(badOptions)) {
		for(int i=0; i < badOptions.size(); ++i)
			std::cerr << "ERROR: Got unexpected number of arguments for option " << badOptions[i] << ".\n\n";
			
		opt.getUsage(usage);
		std::cout << usage;
		return 1;
	}

	std::vector< std::vector<double> > pts;
	opt.get("-p")->getMultiDoubles(pts);
	for(int j=0; j < pts.size(); ++j)
		std::cout << "Point " << j << ": " << pts[j][0] << " " << pts[j][1] << " " << pts[j][2] << std::endl;

	return 0;
}