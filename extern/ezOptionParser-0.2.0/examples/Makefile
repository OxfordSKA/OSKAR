.PHONY: test clean

# Windows.
#CXX=cl
#CXXFLAGS=/EHsc /O2 /I.. /Fo
#EXT=.exe

# GNU/Linux.
CXX=g++
CXXFLAGS=-O2 -I.. -o
EXT=

BIN=long$(EXT) complete$(EXT) fileio$(EXT) full$(EXT) multi$(EXT) parseindex$(EXT) pretty$(EXT) short$(EXT) usage$(EXT) valid$(EXT) validfast$(EXT) validrange$(EXT) validrangefast$(EXT)
SRCS=long complete fileio full multi parseindex pretty short usage valid validfast validrange validrangefast

all: test

test: $(BIN)
	bash test.sh
	
complete$(EXT): ../ezOptionParser.hpp multi.cpp
	$(CXX) complete.cpp $(CXXFLAGS) complete$(EXT)

fileio$(EXT): ../ezOptionParser.hpp fileio.cpp
	$(CXX) fileio.cpp $(CXXFLAGS) fileio$(EXT)

full$(EXT): ../ezOptionParser.hpp full.cpp
	$(CXX) full.cpp $(CXXFLAGS) full$(EXT)

long$(EXT): ../ezOptionParser.hpp long.cpp
	$(CXX) long.cpp $(CXXFLAGS) long$(EXT)

multi$(EXT): ../ezOptionParser.hpp multi.cpp
	$(CXX) multi.cpp $(CXXFLAGS) multi$(EXT)

parseindex$(EXT): ../ezOptionParser.hpp parseindex.cpp
	$(CXX) parseindex.cpp $(CXXFLAGS) parseindex$(EXT)

pretty$(EXT): ../ezOptionParser.hpp pretty.cpp
	$(CXX) pretty.cpp $(CXXFLAGS) pretty$(EXT)

short$(EXT): ../ezOptionParser.hpp short.cpp
	$(CXX) short.cpp $(CXXFLAGS) short$(EXT)

usage$(EXT): ../ezOptionParser.hpp usage.cpp
	$(CXX) usage.cpp $(CXXFLAGS) usage$(EXT)

valid$(EXT): ../ezOptionParser.hpp valid.cpp
	$(CXX) valid.cpp $(CXXFLAGS) valid$(EXT)

validfast$(EXT): ../ezOptionParser.hpp validfast.cpp
	$(CXX) validfast.cpp $(CXXFLAGS) validfast$(EXT)

validrange$(EXT): ../ezOptionParser.hpp validrange.cpp
	$(CXX) validrange.cpp $(CXXFLAGS) validrange$(EXT)

validrangefast$(EXT): ../ezOptionParser.hpp validrangefast.cpp
	$(CXX) validrangefast.cpp $(CXXFLAGS) validrangefast$(EXT)

html: 
	$(foreach var,$(SRCS),pygmentize -O full,linenos=1,style=manni -o ../html/$(var).html $(var).cpp;)
	
memtest: $(BIN)
	./memtest.sh

clean:
	-rm -f *.obj *.exe *.o $(BIN) *.stackdump core* tmp.* *.html *~
