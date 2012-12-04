/*
20110505 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
	ezOptionParser opt;

	opt.overview = "Demo of automatic usage message creation.";
	opt.syntax = "usage [OPTIONS]";
	opt.example = "usage -h\n\n";
	opt.footer = "usage v0.1.4 Copyright (C) 2011 Remik Ziemlinski\nThis program is free and without warranty.\n";

	opt.add(
		"0", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Display usage instructions.\nThere is a choice of three different layouts for description alignment. Your choice can be any one of the following to suit your style:\n\n0 - align (default)\n1 - interleave\n2 - stagger", // Help description.
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
		"Simple flag with a very long help description that will be split automatically into a two column format when usage is printed for this program. Newlines will also help with justification.", // Help description.
		"-f",     // Flag token. 
		"-flg",   // Flag token.
		"--flag" // Flag token.
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

	return 0;
}