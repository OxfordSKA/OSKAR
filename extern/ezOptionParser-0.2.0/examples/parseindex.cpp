/*
20121120 rsz Created.
*/
#include <stdio.h>
#include "ezOptionParser.hpp"

using namespace ez;

int main(int argc, const char * argv[]) {
  ezOptionParser opt;

  opt.overview = "Demo of parse index for options.";
  opt.syntax = "parseindex [OPTIONS] in out";
  opt.example = "./parseindex skip -f foo -c bar,baz -c bam -f ooz -c yum ignore";
  
  opt.add(
    "", // Default.
    0, // Required?
    -1, // Number of args expected.
    ',', // Delimiter if expecting multiple args.
    "Some list of config parameters.", // Help description.
    "-c" // Flag token.
  );

  opt.add(
    "", // Default.
    0, // Required?
    1, // Number of args expected.
    0, // Delimiter if expecting multiple args.
    "Some string.", // Help description.
    "-f" // Flag token.
  );

  opt.parse(argc, argv);

  int i,j,n;
  const char * flags[] = {"-c", "-f"};
  
  for(i=0; i < 2; ++i) {
    if (opt.isSet(flags[i])) {
      std::vector<int> & indices = opt.get(flags[i])->parseIndex;
      n = indices.size();
      std::cout << flags[i] << " indices: ";
      
      for(j=0; j < n; ++j)
        std::cout << indices[j] << " "; 
      
      std::cout << std::endl;
    }
  }

  return 0;
}