#!/usr/bin/env python

''' 1. annotate a forest with local features (node-local or edge-local)
    2. split nodes by annotating parent/other info.
'''

import sys
import time

from xrstree import XRSTree
from features import *
from forest import Forest, get_weights
from utility import *
#import heads

from svector import Vector

debug = False

def print_fnames(fs):
    if len(fs) > 0:
        print "\n".join(["\n".join((name,) * 1) for name in fs])  ## onecount used to return (feat, 1), but now only feat


def local_feats(forest, fclasses, oracleedges=None):

    for node in forest:
        ## you will have to annotate parentlabel and heads as soon as you get a tree node by assembling
        node.parentlabel = None        
        
        if not node.is_spurious():  ## neglect spurious nodes
            nodefvector = Vector()

            if hasattr(node, "same"):
                pass
                ##nodefvector = node.same.extrafvector
            else:
                for feat in fclasses:
                    if feat.is_nodelocal():
                        fs = feat.extract(node, forest.sent)
                        if opts.extract:
                            print_fnames(fs) 
                        else:
                            for f in fs:
                                nodefvector[f] += 1 #Vector.convert_fullname(fs)        

                node.fvector += nodefvector
##            print >> logs, "%s -------\t%s" % (node, node.fvector)

            for edge in node.edges:
                
                if debug:
                    print >> logs, "--------", edge.shorter()                    

                if hasattr(edge, "same"):
                    edgefvector = edge.same.extrafvector
                else:
                    node.subs = edge.subs
##                    node.rehash()  ## reset coordination and heads
                    
                    edgefvector = Vector()
                    for feat in fclasses:
                        if feat.is_edgelocal():
                            fs = feat.extract(node, forest.sent)
                            if opts.extract:
                                print_fnames(fs)
                            else:
                                for f in fs:
                                    edgefvector[f] += 1 #Vector.convert_fullname(fs)

                    if opts.xrsrule:
                        if edge.ruleid in forest.rules:
                            rulestr = forest.rules[edge.ruleid]
                            edgefvector["XRS~" + "_".join(rulestr.split())] =  1
                        else:
                            print >> logs, "BAD RULE ID: %s" %  edge.ruleid
                        
                    if opts.cfgrule:
                        if edge.ruleid in forest.rules:
                            rulestr = forest.rules[edge.ruleid]
                            if rulestr not in ruleobj_cache:
                                try:
                                    # "S -> A B" => "S->A_B"
                                    ruleobj = XRSTree.parse(rulestr).cfgrules()
                                except:
                                    print >> logs, "BAD", rulestr
                                ruleobj_cache[edge.ruleid] = ruleobj
##                                forest.rules[edge.ruleid] = (rulestr, ruleobj)
                            else:
                                ruleobj = ruleobjcache[edge.ruleid]

                            for cfgrule in ruleobj:
                                edgefvector["CFG~" + cfgrule] = 1
                                
                        else:
                            print >> logs, "BAD RULE ID: %s" %  edge.ruleid

                    if opts.cheat_oracle:
                        if edge in oracleedges:
                            edgefvector["ORACLE"] = 1
                        else:
                            edgefvector["NOT_ORACLE"] = 1

                edge.extrafvector = edgefvector
                edge.fvector += edgefvector
                if debug and len(edge.fvector) > 1:
                    print >> logs, "%s ---------\t%s" % (edge, edge.fvector.pp(usename=True))
    

if __name__ == "__main__":

##    sys.setrecursionlimit(20)
    
    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog [options (-h)] [<feats>]")
    optparser.add_option("-s", "--suffix", dest="suffix", help="dump suffix (1.suffix)", metavar="SUF")
    optparser.add_option("-q", "--quiet", dest="quiet", action="store_true", help="no dumping", default=False)
    optparser.add_option("-d", "--debug", dest="debug", action="store_true", help="show debug", default=False)
    optparser.add_option("-e", "--extract", dest="extract", action="store_true", \
                         help="extract features names", default=False)
    optparser.add_option("", "--cheat", dest="cheat_oracle", action="store_true", \
                         help="cheat oracle feature", default=False)
    optparser.add_option("-w", "--weights", dest="weights", type=str, help="weights file or str", metavar="WEIGHTS", default="") # empty weight
    optparser.add_option("", "--cfgrule", dest="cfgrule", action="store_true", \
                 help="PCFG rule feature (edge-local)", default=False)
    optparser.add_option("", "--xrsrule", dest="xrsrule", action="store_true", \
                 help="XRS rule feature (edge-local)", default=False)

    (opts, args) = optparser.parse_args()
    debug = opts.debug
    weights = get_weights(opts.weights)

    fclasses = prep_features(args, read_names=(not opts.extract))
    print >> logs, "features classes", fclasses

    start = time.time()

    ruleobj_cache = {}
    
    for i, forest in enumerate(Forest.load("-"), 1):

        if forest is not None:

            oracleedges = None
            if opts.cheat_oracle:
                bleu, hyp, fv, edgelist = forest.compute_oracle(weights, 0) ## absolute oracle
                oracleedges = set(edgelist)

            local_feats(forest, fclasses, oracleedges)
            if not opts.quiet and not opts.extract:
                if opts.suffix is not None:
                    forest.dump(open("%d.%s" % (i, opts.suffix), "wt"))
                else:
                    forest.dump()

        else:
            print 

    total_time = time.time() - start
    print >> logs, "overall: %d sents. local features extracted in %.2lf secs (avg %.2lf per sent)" % \
          (i, total_time, total_time/(i))
