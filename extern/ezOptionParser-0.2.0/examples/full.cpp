/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Full demo of all the features.";
	opt.syntax = "full [OPTIONS]";
	opt.example = "full -h\n\n";
	opt.footer = "full v0.1.4 Copyright (C) 2011 Remik Ziemlinski\nThis program is free and without warranty.\n";

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Print this usage message in one of three different layouts. The choices are:\n0 - aligned (default)\n1 - interleaved\n2 - staggered", // Help description.
		"-h",     // Flag token. 
		"-help", // Flag token.
		"--help", // Flag token.
		"--usage" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		0, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Do not print all input arguments for test. By default, all the inputs will be pretty printed to show which category they belong to (first, options, last) and what their values are. This shows that flag names can be arbitrary and don't need to begin with a single or double dash (-,--).", // Help description.
		"+d",     // Flag token. 
		"+dbg",  // Flag token.
		"+debug" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Test integer input.", // Help description.
		"-n",     // Flag token. 
		"-int", 	// Flag token.
		"--int", 	// Flag token.
		"--integer" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Test long input.", // Help description.
		"-l",     // Flag token. 
		"-lng", 	// Flag token.
		"-long", 	// Flag token.
		"--long" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Test unsigned long input.", // Help description.
		"-u",     // Flag token. 
		"-ulong", // Flag token.
		"--ulong" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Test float input.", // Help description.
		"-f",     // Flag token. 
		"-flt", 	// Flag token.
		"-float", // Flag token.
		"--float" // Flag token.
	);

	opt.add(
		"0", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Test double input.", // Help description.
		"-d",     // Flag token. 
		"-dbl", 	// Flag token.
		"-double", // Flag token.
		"--double" // Flag token.
	);

	opt.add(
		"A default string.", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Test string input.", // Help description.
		"-s",     // Flag token. 
		"-str", 	// Flag token.
		"-string", 	// Flag token.
		"--string" // Flag token.
	);

	opt.add(
		"0,1,2,3,4,5,6,7,8,9,10", // Default.
		0, // Required?
		1, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Test integer list input delimited with comma.", // Help description.
		"-nl",     // Flag token. 
		"-nlist", 	// Flag token.
		"-intlist", 	// Flag token.
		"--intlist" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		':', // Delimiter if expecting multiple args.
		"Test long list input delimited by colon.", // Help description.
		"-ll",     // Flag token. 
		"-llist", 	// Flag token.
		"-longlist", 	// Flag token.
		"--longlist" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Test unsigned long list input.", // Help description.
		"-ul",     // Flag token. 
		"-ulist", 	// Flag token.
		"-ulonglist", 	// Flag token.
		"--ulonglist" // Flag token.
	);

	opt.add(
		"-2.1e-20,-.2,-.001,0,1,1e10", // Default.
		0, // Required?
		1, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Test float list input.", // Help description.
		"-fl",     // Flag token. 
		"-flist", 	// Flag token.
		"-floatlist", 	// Flag token.
		"--floatlist" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Test double list input.", // Help description.
		"-dl",     // Flag token. 
		"-dlist", 	// Flag token.
		"-dbllist", 	// Flag token.
		"--doublelist" // Flag token.
	);

	opt.add(
		"\"string list item1\",item2,\"list # item3\",2000/1/1", // Default.
		0, // Required?
		1, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Test string list input.", // Help description.
		"-sl",     // Flag token. 
		"-slist", 	// Flag token.
		"-strlist", 	// Flag token.
		"--strlist" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Import additional arguments from file. Multiple files can be imported by using multiple instances of this option. The file options will add to those set on the command line. If you want them to overwrite the command line options, then use +import.", // Help description.
		"-i",     // Flag token. 
		"-import" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Import additional arguments from file. Multiple files can be imported by using multiple instances of this option. The last file will overwrite all options set previously by either command line or another file. If you want them to add to previously set options, then use -import.", // Help description.
		"+i",     // Flag token. 
		"+import" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Export only set arguments to file.", // Help description.
		"-e",     // Flag token. 
		"-export" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Export all arguments, including defaults, to file.", // Help description.
		"+e",     // Flag token. 
		"+export" // Flag token.
	);

	opt.parse(argc, argv);

	if (opt.isSet("-h")) {
		std::string usage;
		int layout;
		opt.get("-h")->getInt(layout);
		switch(layout) {
		case 0:
			opt.getUsage(usage,80,ezOptionParser::ALIGN);
			break;
		case 1:
			opt.getUsage(usage,80,ezOptionParser::INTERLEAVE);
			break;
		case 2:
			opt.getUsage(usage,80,ezOptionParser::STAGGER);
			break;
		}
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

	if (opt.isSet("+i")) {
		// Import one or more files that use # as comment char.
		std::vector< std::vector<std::string> > files;
		opt.get("+i")->getMultiStrings(files);

		if(!files.empty()) {
			std::string file = files[files.size()-1][0];
			opt.resetArgs();
			if (! opt.importFile(file.c_str(), '#'))
				std::cerr << "ERROR: Failed to open file " << file << std::endl;
		}
	}

	if (opt.isSet("-e")) {
		std::string outfile;
		opt.get("-e")->getString(outfile);
		opt.exportFile(outfile.c_str(), false);
	}

	if (opt.isSet("+e")) {
		std::string outfile;
		opt.get("+e")->getString(outfile);
		opt.exportFile(outfile.c_str(), true);
	}

	if (!opt.isSet("+d")) {
		std::string pretty;
		opt.prettyPrint(pretty);
		std::cout << pretty;
	}

	return 0;
}