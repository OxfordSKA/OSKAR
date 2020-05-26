#!/bin/bash

hdiutil $@
while [ $? -ne 0 ]; do
    sleep 1
    hdiutil $@
done
