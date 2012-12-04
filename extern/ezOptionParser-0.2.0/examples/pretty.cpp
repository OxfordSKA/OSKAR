/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Demo of pretty printing everything parsed for debugging.";
	opt.syntax = "pretty [OPTIONS]";
	opt.example = "pretty foo bar --print -fake --dummy -list 1:2:4:8:16 in1 in2 in3 out\n";

	opt.add(
		"", // Default.
		0, // Required?
		0, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Print all inputs and their category.", // Help description.
		"-p",     // Flag token. 
		"-prn",   // Flag token.
		"--print" // Flag token.
	);

	opt.add(
		"", // Default.
		0, // Required?
		-1, // Number of args expected.
		':', // Delimiter if expecting multiple args.
		"Colon delimited tuple of any length.", // Help description.
		"-l", // Flag token.
		"-lst", // Flag token.
		"-list", // Flag token.
		"--list" // Flag token.
	);

	opt.parse(argc, argv);

	if (opt.isSet("-p")) {
		std::string pretty;
		opt.prettyPrint(pretty);
		std::cout << pretty;
	}

	std::vector<int> list;
	opt.get("-lst")->getInts(list);
	std::cout << "\nList:";
	for(int j=0; j < list.size(); ++j)
		std::cout << " " << list[j];
		
	std::cout << std::endl;

	return 0;
}