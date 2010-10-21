#!/bin/sh

grep -P "^\t" | perl -ple 's/\|\|\|.*$//g' \
| awk '{for (i=1;i<=NF;i++) print $i}' \
| grep -P ^\" | perl -ple 's/^\"//g;s/\"$//g;s/\\//g' \
| sort | uniq
