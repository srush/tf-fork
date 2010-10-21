#!/usr/bin/env python
# bleu+1.py
# 4 Apr 2006
# David Chiang

import sys, itertools, math
import optparse

import bleu

# turn off NIST normalization
#bleu.nist_tokenize = False

# lhuang: difference b/w bleu+1 and bleu:
# 1) bleu+1 uses +1 smoothing (see below)
# 2) by default, bleu+1 does not include brevity penalty
# 3) effective ref_length might be different

# N.B.! have to be consistent
#bleu.eff_ref_len = "closest"

# usage: bleu+1.py <test> <ref>+

def score_single_cooked(comps, brevity=False, n=4):
    if comps['testlen'] == 0:
        return 0
    logbleu = 0.0
    for k in xrange(n):
        # lhuang: +1 smoothing, so that precision won't be zero.
        logbleu += math.log(comps['correct'][k]+1)-math.log(comps['guess'][k]+1)
        #sys.stderr.write("%d/%d " % (comps['correct'][k], comps['guess'][k]))
    logbleu /= float(n)

    if brevity:
        logbleu += min(0,1-(float(comps['reflen']+1))/(comps['testlen']+1))

    return math.exp(logbleu)

if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-m", "--map-file", dest="mapfilename", help="map file indicating sentence number in reference set for each line of input")
    optparser.add_option("-b", "--brevity-penalty", dest="brevitypenalty", action="store_true", help="assess brevity penalty")
    (opts, args) = optparser.parse_args()

    
    n = 4

    cookedrefs = []
    for lines in itertools.izip(*[file(filename) for filename in args[1:]]):
        cookedrefs.append(bleu.cook_refs(lines, n=n))

    if opts.mapfilename is not None:
        linemap = []
        for line in file(opts.mapfilename):
            linemap.append(int(line))
    else:
        linemap = range(len(cookedrefs))

    if args[0] == "-":
        infile = sys.stdin
    else:
        infile = open(args[0])
    test1 = []
    for (line,i) in itertools.izip(infile, linemap):
        test1.append(bleu.cook_test(line, cookedrefs[i], n=n))

    total = 0.
    n_sent = 0

    for comps in test1:

        score = score_single_cooked(comps)
        sys.stdout.write("bleu+1=%f\n" % score)
        total += score
        n_sent += 1

    sys.stderr.write("average: %s\n" % (total/n_sent))
    
