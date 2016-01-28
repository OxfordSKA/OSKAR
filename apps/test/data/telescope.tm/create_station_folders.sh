#!/bin/bash
#
# Script to create station folders by creating numbered folders each with a
# copy of the antenna_layout.txt file.
#

for i in {1..30}; do
    name=$(printf "station%03i" $i)
    if [ ! -d $name ]; then
        mkdir $name
    fi
    if [ ! -e ${name}/layout.txt ]; then
        cp antenna_layout.txt ${name}/layout.txt
    fi
done
