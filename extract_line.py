#!/usr/bin/env python

import sys
import fileinput

if __name__ == "__main__":
    import optparse
    optparser = optparse.OptionParser(usage="usage -n sentNUMfile -t tree-file")
    optparser.add_option("-n", dest="nfile", type=str, default=None)
    optparser.add_option("-t", dest="tfile", type=str, default=None)

    opts, args = optparser.parse_args()
    nset = set([int(line) for line in fileinput.input(opts.nfile)])

    print "".join([line for i, line in enumerate(fileinput.input(opts.tfile), 1) if i in nset ])
