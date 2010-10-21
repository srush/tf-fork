#!/bin/bash
# doesn't work on mac!

i=$1
sed -e 's/^$/\$\$\$/g' | tr "\n" "\v" | sed -e 's/\$\$\$\v/\n/g' | sed -n ${i}p | tr "\v" "\n" 
