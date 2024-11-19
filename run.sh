#!/bin/bash

echo "Adding some modules"
module add gcc-13.1
module add cmake

echo "#################"
echo "    COMPILING    "
echo "#################"

make nn

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
nice -n 19 ./bin/nn
