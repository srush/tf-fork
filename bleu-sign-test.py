#!/usr/bin/env python
# bleu-sign-test.py
# 4 Apr 2006
# David Chiang

import sys, itertools, math
sys.path.append("/home/nlg-01/chiangd/hiero")

import bleu

def normalize(words):
    return words

#bleu.normalize = normalize

# usage: bleu-sign-test.py <test1> <test2> <ref>+

verbose = False

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

    score1 = bleu.score_cooked(test1)
    print "System 1: %f" % score1
    print "System 2: %f" % bleu.score_cooked(test2)

    better = worse = 0
    fake = test1[:]
    for i in xrange(len(fake)):
        fake[i] = test2[i]

        fake_score = bleu.score_cooked(fake)
        if fake_score > score1:
            better += 1
        elif fake_score < score1:
            worse += 1

        if verbose:
            print "sent %d %s %s" % (i, score1, fake_score)
        
        fake[i] = test1[i]

    print "System 2 was better on %d sentences, worse on %d sentences" % (better, worse)

    n = float(better+worse)
    mean = float(better)/n
    se = math.sqrt(mean*(1-mean)/n)

    print "Pr(better|different) = %f" % mean
    print "95%% confidence interval: (%f,%f)" % (mean-1.96*se,mean+1.96*se)
    print "99%% confidence interval: (%f,%f)" % (mean-2.58*se,mean+2.58*se)

    if mean-2.58*se > 0.5:
        print "System 2 is significantly better (p < 0.01)"
    elif mean+2.58*se < 0.5:
        print "System 2 is significantly worse (p < 0.01)"
    elif mean-1.96*se > 0.5:
        print "System 2 is significantly better (p < 0.05)"
    elif mean+1.96*se < 0.5:
        print "System 2 is significantly worse (p < 0.05)"
    else:
        print "No significant difference"
    
