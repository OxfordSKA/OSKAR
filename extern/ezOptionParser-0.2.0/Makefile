.PHONY: all examples html install memtest test clean

APP = ezOptionParser
DEVDIR = ezoptionparser-code
PREFIX ?= /usr/local
export PREFIX

all: examples

examples:
	cd examples && $(MAKE) all

clean:
	cd examples && $(MAKE) clean
	-rm -f *~

html:
	cd examples && $(MAKE) html
	pygmentize -O full,linenos=1,style=manni -o html/ezOptionParser.html ezOptionParser.hpp
	markdown README.md > html/index.html
	cp ezOptionParser.hpp html

install:
	cp ezOptionParser.hpp $(PREFIX)/include

memtest:
	cd examples && $(MAKE) memtest

test:
	cd examples && $(MAKE) test

dist:
  ifndef VER
		@echo "ERROR: VER is not defined. Try: make dist VER=0.1.0"
  else
		cd ..; rm -fr $(APP)-$(VER); mkdir $(APP)-$(VER); rsync --cvs-exclude -a $(DEVDIR)/ $(APP)-$(VER); tar zcvf $(APP)-$(VER).tar.gz $(APP)-$(VER); rm -fr $(APP)-$(VER);
  endif
