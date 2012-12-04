/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

void Usage(ezOptionParser& opt) {
	std::string usage;
	opt.getUsage(usage, 80, ezOptionParser::ALIGN);
	std::cout << usage;
};

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Demo of parser's features.";
	opt.syntax = "complete first second [OPTIONS] in1 [... inN] out";
	opt.example = "complete a b -f --list 1,2,3 --list 4,5,6,7,8 -s string -int -2147483648,2147483647 -ulong 9223372036854775807 -float 3.40282e+038 -double 1.79769e+308 f1 f2 f3 f4 f5 f6 fout\n\n";
	opt.footer = "ezOptionParser 0.1.4  Copyright (C) 2011 Remik Ziemlinski\nThis program is free and without warranty.\n";

	opt.add(
		"", // Default.
		0, // Required?
		0, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Display usage instructions",  // Help description.
		"-h",     // Flag token. 
		"-help",  // Flag token.
		"--help", // Flag token.
		"--usage" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		0, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Simple flag with a very long help description that will be split automatically into a two column format when usage is printed for this program. Newlines will also help with justification.\nFor example:\n0 - an item\n1 - another item\n2 - and another item", // Help description.
		"-f",     // Flag token. 
		"-flg",   // Flag token.
		"--flag" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		-1, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Lists of arbitrary lengths.", // Help description.
		"-l",    // Flag token. 
		"-lst",  // Flag token.
		"-list",  // Flag token.
		"--list" // Flag token.
	);

	opt.add(
		"hello", // Default.
		1, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Single string.", // Help description.
		"-s", // Flag token.
		"-str", // Flag token.
		"-string", // Flag token.
		"--string" // Flag token.
	);

	opt.add(
		"0,1", // Default.
		0, // Required?
		2, // Number of args expected.
		',', // Delimiter if expecting multiple args.
		"Integer placeholder.", // Help description.
		"-i", // Flag token.
		"-int", // Flag token.
		"-integer", // Flag token.
		"--integer" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Unsigned long placeholder.", // Help description.
		"-ul", // Flag token.
		"-ulong" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Float placeholder.", // Help description.
		"-float" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Double placeholder.", // Help description.
		"-double" // Flag token.
	);

	opt.parse(argc, argv);

	if (opt.isSet("-h")) {
		Usage(opt);
		return 1;
	}

	if (opt.lastArgs.size() < 2) {
		std::cerr << "ERROR: Expected at least 2 arguments.\n\n";
		Usage(opt);
		return 1;
	} 

	std::vector<std::string> badOptions;
	int i;
	if(!opt.gotRequired(badOptions)) {
		for(i=0; i < badOptions.size(); ++i)
			std::cerr << "ERROR: Missing required option " << badOptions[i] << ".\n\n";
		Usage(opt);
		return 1;
	}

	if(!opt.gotExpected(badOptions)) {
		for(i=0; i < badOptions.size(); ++i)
			std::cerr << "ERROR: Got unexpected number of arguments for option " << badOptions[i] << ".\n\n";
			
		Usage(opt);
		return 1;
	}

	std::string firstArg;
	if (opt.firstArgs.size() > 0)
		firstArg = *opt.firstArgs[0];
		
	bool flag = false;
	if (opt.isSet("-f")) {
		flag = true;
	}

	std::vector<int> list;
	opt.get("-lst")->getInts(list);
	std::cout << "\nList:";
	for(int j=0; j < list.size(); ++j)
		std::cout << " " << list[j];
		
	std::cout << std::endl;

	return 0;
}
