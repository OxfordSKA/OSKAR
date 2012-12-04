/*
20111010 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

void Usage(ezOptionParser& opt) {
  std::string usage;
  opt.getUsage(usage);
  std::cout << usage;
};

int main(int argc, const char * argv[]) {
  ezOptionParser opt;

  opt.overview = "Demo of validation options.";
  opt.syntax = "validrange [OPTIONS]";
  opt.example = "validrange -i validin2.txt -o tmp\n\n";

  opt.add(
    "", // Default.
    0, // Required?
    0, // Number of args expected.
    0, // Delimiter if expecting multiple args.
    "Display usage instructions.", // Help description.
    "-h",     // Flag token. 
    "--help", // Flag token.
    "--usage" // Flag token.
  );

  ezOptionValidator* vS1 = new ezOptionValidator("s1", "lt", "-10");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Signed byte (aka char, 1 byte) < -10.", // Help description.
    "-s1",     // Flag token.
    vS1
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of signed byte (aka char, 1 byte) < -10.", // Help description.
    "-s1list",     // Flag token.
    vS1
  );

  ezOptionValidator* vU1 = new ezOptionValidator("u1", "le", "100");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Unsigned byte (aka unsigned char, 1 byte) <= 100.", // Help description.
    "-u1",     // Flag token.
    vU1
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of unsigned byte (aka unsigned char, 1 byte) <= 100.", // Help description.
    "-u1list",     // Flag token.
    vU1
  );

  ezOptionValidator* vS2 = new ezOptionValidator("s2", "ge", "-3000");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Short (2 bytes) >= -3000.", // Help description.
    "-s2",     // Flag token.
    vS2
  );
  
  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of short (2 bytes) >= -3000.", // Help description.
    "-s2list",     // Flag token.
    vS2
  );

  ezOptionValidator* vU2 = new ezOptionValidator("u2", "gt", "10000");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Unsigned short (2 bytes) > 10000.", // Help description.
    "-u2",     // Flag token.
    vU2
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of unsigned short (2 bytes) > 10000.", // Help description.
    "-u2list",     // Flag token.
    vU2
  );

  ezOptionValidator* vS4 = new ezOptionValidator("s4", "gtlt", "-2000000001,-1111111110");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Integer (4 bytes) -2000000001 < x < -1111111110.", // Help description.
    "-s4",     // Flag token.
    vS4
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of integer (4 bytes)  -2000000001 < x < -1111111110.", // Help description.
    "-s4list",     // Flag token.
    vS4
  );

  ezOptionValidator* vU4 = new ezOptionValidator("u4", "gelt", "0,4294967291");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Unsigned int (4 bytes) 0 <= and < 4294967291.", // Help description.
    "-u4",     // Flag token.
    vU4
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of unsigned int (4 bytes) 0 <= and < 4294967291.", // Help description.
    "-u4list",     // Flag token.
    vU4
  );
  
  ezOptionValidator* vU8 = new ezOptionValidator("u8", "gele", "0,18446744073709551615");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Unsigned long long (8 bytes) 0 <= and <= 18446744073709551615.", // Help description.
    "-u8",     // Flag token.
    vU8
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of unsigned long long (8 bytes) 0 <= and <= 18446744073709551615.", // Help description.
    "-u8list",     // Flag token.
    vU8
  );

  ezOptionValidator* vS8 = new ezOptionValidator("s8", "gtle", "-9223372036854775801,-8888888888888888888");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Long long (68 bytes) -9223372036854775801 < and <= -8888888888888888888.", // Help description.
    "-s8",     // Flag token.
    vS8
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of long long (68 bytes) between -9223372036854775808 and 9223372036854775807.", // Help description.
    "-s8list",     // Flag token.
    vS8
  );

  ezOptionValidator* vF = new ezOptionValidator("f", "in", "-3.40282e+038,-3.2e+038,-3e+038");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Float (4 bytes) that is either -3.40282e+038,-3.2e+038,-3e+038.", // Help description.
    "-f",     // Flag token.
    vF
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of float (4 bytes) that is either -3.40282e+038,-3.2e+038,-3e+038.", // Help description.
    "-flist",     // Flag token.
    vF
  );

  ezOptionValidator* vD = new ezOptionValidator("d", "gtlt", "1e307,2e308");
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Double (8 bytes) 1e307 < and < 2e308.", // Help description.
    "-d",     // Flag token.
    vD
  );

  opt.add(
    "0", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "List of double (8 bytes) 1e307 < and < 2e308.", // Help description.
    "-dlist",     // Flag token.
    vD
  );
  
  ezOptionValidator* vBool = new ezOptionValidator("t", "in", "0,1,n,y,f,t,no,yes,false,true", false);
  
  opt.add(
    "0", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Boolean string. Either one of these (case sensitive):\n0,1,n,y,f,t,no,yes,false,true", // Help description.
    "-b",     // Flag token.
    vBool
  );

  opt.add(
    "", // Default.
    0, // Required?
    1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "File with input options to aid stress testing.", // Help description.
    "-i" // Flag token.
  );

	opt.add(
		"", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"File to export options.", // Help description.
		"-o"     // Flag token. 
	);

  opt.parse(argc, argv);

  if (opt.isSet("-h")) {
    Usage(opt);
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

  std::vector<std::string> badArgs;
  if(!opt.gotValid(badOptions, badArgs)) {
    for(i=0; i < badOptions.size(); ++i)
      std::cerr << "ERROR: Got invalid argument \"" << badArgs[i] << "\" for option " << badOptions[i] << ".\n\n";
      
    //Usage(opt);
    return 1;
  }
  
	if (opt.isSet("-o")) {
		std::string file;
		opt.get("-o")->getString(file);
		// Exports all options if second param is true; unset options will just use their default values.
		opt.exportFile(file.c_str(), false);
	}

  return 0;
}