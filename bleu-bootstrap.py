#!/usr/bin/env python

import sys
import bleu
bleu.eff_ref_len = 'closest'
import random, itertools

if __name__ == "__main__":
    import getopt

    (opts,args) = getopt.getopt(sys.argv[1:], "rctpv", [])
    for (opt,parm) in opts:
        if opt == "-c":
            bleu.preserve_case = True
        elif opt == "-t":
            bleu.nist_tokenize = False
        elif opt == "-p":
            bleu.clip_len = True
        elif opt == "-v":
            verbose = True

    test1 = []
    test2 = []
    for lines in itertools.izip(*[file(filename) for filename in args]):
        cookedrefs = bleu.cook_refs(lines[2:])
        test1.append(bleu.cook_test(lines[0], cookedrefs))
        test2.append(bleu.cook_test(lines[1], cookedrefs))

    print "System 1: %f" % bleu.score_cooked(test1)
    print "System 2: %f" % bleu.score_cooked(test2)

    better1 = better2 = 0

    n = 1000

    diffs = []
    for i in xrange(n):
        fake1 = []
        fake2 = []
        for j in xrange(len(test1)):
            r = random.randrange(len(test1))
            fake1.append(test1[r])
            fake2.append(test2[r])

        score1 = bleu.score_cooked(fake1)
        score2 = bleu.score_cooked(fake2)

        if score1 > score2:
            better1 += 1
        elif score2 > score1:
            better2 += 1

    print "System 1 was better on %d samples, System 2 on %d samples" % (better1, better2)

    
