/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Demo of file import and export.";
	opt.syntax = "fileio [OPTIONS]";
	opt.example = "fileio -i in1.txt -i in2.txt -o out.txt\n\n";

	opt.add(
		"", // Default.
		0, // Required?
		0, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Print this usage message.", // Help description.
		"-h",     // Flag token. 
		"-help", // Flag token.
		"--help", // Flag token.
		"--usage" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Test string input.", // Help description.
		"-s",     // Flag token. 
		"-str", // Flag token.
		"--string" // Flag token.
	);

	opt.add(
		"in.txt", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"File to import arguments.", // Help description.
		"-i",     // Flag token. 
		"--import" // Flag token.
	);

	opt.add(
		"out.txt", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"File to export arguments.", // Help description.
		"-o",     // Flag token. 
		"--export" // Flag token.
	);

	opt.parse(argc, argv);

	std::string usage;
	std::vector<std::string> badOptions;

	if(!opt.gotExpected(badOptions)) {
		for(int i=0; i < badOptions.size(); ++i)
			std::cerr << "ERROR: Got unexpected number of arguments for option " << badOptions[i] << ".\n\n";
			
		opt.getUsage(usage);
		std::cout << usage;
		return 1;
	}

	if (opt.isSet("-h")) {
		opt.getUsage(usage);
		std::cout << usage;
		return 1;
	}

	if (opt.isSet("-i")) {
		// Import one or more files that use # as comment char.
		std::vector< std::vector<std::string> > files;
		opt.get("-i")->getMultiStrings(files);

		for(int j=0; j < files.size(); ++j)
			if (! opt.importFile(files[j][0].c_str(), '#'))
				std::cerr << "ERROR: Failed to open file " << files[j][0] << std::endl;
	}

	if (opt.isSet("-o")) {
		std::string file;
		opt.get("-o")->getString(file);
		// Exports all options if second param is true; unset options will just use their default values.
		opt.exportFile(file.c_str(), true);
	}

	std::string pretty;
	opt.prettyPrint(pretty);
	std::cout << pretty;

	return 0;
}