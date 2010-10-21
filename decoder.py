#!/usr/bin/env python

import sys
import time
import heapq

logs = sys.stderr

from forest import Forest, get_weights
from svector import Vector
#from oracle import forest_oracle
#from readkbest import NBestForest
#from features import *
#from local_feat import *
#from parseval import *
#from features import read_features
from node_and_hyperedge import Hyperedge
from bleu import Bleu

from collections import defaultdict

class Decoder(object):

    ## to be used by size_ratiox
    MAX_NUM_BRACKETS = -20.0 #ratio = 1
    
    def __init__(self):
        self.reset()

    def load_time(self):
        return Forest.load_time
    
    def reset(self):
        self.decode_time = 0
        self.oracle_time = 0
        self.extract_time = 0

    def decode(self, forest, weights):
        ''' a wrapper. returns tuple (score, fvector, tree) '''
        self.decode_time -= time.time()
        stuff = self._decode(forest, weights)
        self.decode_time += time.time()
        return stuff

    def fear(self, forest, weights):
        ''' returns a 4-ary tuple (score, oracleparseval, oracletree, oraclefvector). '''        
        bleu_score, hyp, fv, edgelist = forest.compute_oracle(weights, 1, -1) #sc, tr, parseval, edgelist
##            bestscore, besttree, bestfvector, parseval = decoder.decode(forest, weights)        

        mscore = weights.dot(fv)

        b = forest.bleu.copy()
        b.rescore(hyp)
        return mscore, hyp, fv, b
    
    def do_oracle(self, forest, weights):
        ''' returns a 4-ary tuple (score, oracleparseval, oracletree, oraclefvector). '''
        self.oracle_time -= time.time()
        self._oracle(forest, weights)
        self.oracle_time += time.time()

    def load(self, filename):
        ''' returns a generator, yielding a forest at a time. '''
        pass

    def extract_extra(self, forest, all_feats):
        pass

    def set_feats(self, nlfeats):
        self.nlfeats = nlfeats
        print >> logs, "nonlocal=", self.nlfeats
        self.all_feats = nlfeats

    __str__ = lambda x: "Decoder"

    
class LocalDecoder(Decoder):

    def __init__(self, hope=1e-10):
        Decoder.__init__(self)
        self.hope = hope

    def do_decode(self, forest, weights):
        '''to be overridden by BottomUp Decoder'''
        return forest.bestparse(weights, adjust=False)
    
    def _decode(self, forest, weights):
        score, tree, fvector  = self.do_decode(forest, weights)
        if forest.refs != []:
            forest.bleu.rescore(tree)
            parseval = forest.bleu.copy()
        else:
            parseval = Bleu()
        return score, tree, fvector, parseval

    def _oracle(self, forest, weights):
        bleu_score, hyp, fv, edgelist = forest.compute_oracle(weights, self.hope) #sc, tr, parseval, edgelist 
        forest.oracle_tree = hyp
        forest.oracle_fvector = fv
##        if Decoder.MAX_NUM_BRACKETS < 0:
        forest.oracle_size_ratio = 1
        forest.oracle_bleu_score = bleu_score
##        else:
##            forest.oracle_size_ratio = len(tr.all_label_spans()) / Decoder.MAX_NUM_BRACKETS

    def load(self, filenames):
        # new: multiple files
        for filename in filenames.split():
            for forest in Forest.load(filename):
                if forest is not None:
                    yield forest
                else:
                    yield None ## special treatment above

    def extract_extra(self, forest, all_feats):
        local_feats(forest, all_feats)

    __str__ = lambda x: "LocalDecoder"


class BUDecoder(LocalDecoder):
    '''Algorithm 2 (bottom-up) cube pruning decoder.'''

    __str__ = lambda x: "Forest Decoder at k=%d" % (x.k)

    def __init__(self, k, nonlocals=[], check_feats=False, adaptive_base=None):        
        LocalDecoder.__init__(self)
        
        self.k = k
        self.set_feats(nonlocals)
        self.check_feats = check_feats

        if adaptive_base:
            self.adaptive = True
            self.base = adaptive_base   ## smallest adaptive k for spans of length 1
        else:
            self.adaptive = False

    def _oracle(self, forest):
        ''' N.B. need to extract nonlocal features for oracle tree!!!'''
        LocalDecoder._oracle(self, forest)
        self.extract_time -= time.time()
        extra_fv = extract(forest.oracle_tree, forest.sent, self.nlfeats, do_sub=True)        
        forest.oracle_fvector += extra_fv
        self.extract_time += time.time()


    def newgetres(self, edge, vecj, sent, weights):
        '''modified from Hyperedge.getres() '''
        
        fvector = edge.fvector + edge.head.fvector
        score = fvector.dot(weights) ## used to be edge.edge_score 
        subtrees = []
        for i, sub in enumerate(edge.subs):
            if vecj[i] >= len(sub.klist):
                return None
            sc, tr, fv = sub.klist[vecj[i]]
            subtrees.append(tr)
            score += sc
            fvector += fv

        tree = edge.head.assemble(subtrees)
        
        self.extract_time -= time.time()
        nlfvs = extract(tree, sent, self.nlfeats, do_sub=False)
        fvector += nlfvs
        self.extract_time += time.time()

        score += nlfvs.dot(weights)

##        print "--------", edge.head.labelspan(), nlfvs
        
        return score, tree, fvector

    def getcandidates(self, node, sent, weights):
        cand = []
        for edge in node.edges:
            vecone = edge.vecone()
            res = self.newgetres(edge, vecone, sent, weights)
            edge.oldvecs = set([vecone])
            if res is not None:
                cand.append((res, edge, vecone))                    
        heapq.heapify(cand)
        return cand

    def do_decode(self, forest, weights):
        ##score, tree, fvector  = forest.bestparse(weights, adjust=False)

        bestscore = forest.prep_kbest()

        adapted = {}
        sentlen = len(forest.sent)
        for i in range(1, sentlen+1):
            if self.adaptive:
                adapted[i] = int(float(i) / sentlen * (self.k - self.base)) + self.base
            else:
                adapted[i] = self.k
            
##            print >> logs, i, "\t", adapted[i]

        for node in forest:
            if node.is_terminal():
                ################ TERMINAL ###########################
                tree = node.besttree = node.this_tree()
                fvector = node.fvector.copy()

                ## dummy feature extract at terminals, just to set up boundary conditions
                self.extract_time -= time.time()
                nlfvs = extract(tree, forest.sent, self.nlfeats, do_sub=False)
                fvector += nlfvs
                self.extract_time += time.time()

##                assert len(zero_nlfvs) == 0, "%s: %s" % (node, nlfvs)                

                node.beta = node.node_score = fvector.dot(weights)
                node.bestres = (node.node_score, node.besttree, fvector)  ## caution copy

                node.klist = [node.bestres]
                node.kset = set([node.besttree])
                continue

            cand = self.getcandidates(node, forest.sent, weights)
##            print >> logs, node.labelspan(), " degree=", len(node.edges), "\n initial cand\n", "\n".join(map(str, cand))
            
            if cand == []:
                continue            

            templist = []
            for i in xrange(adapted[node.span_width()]):
                if cand == []:
                    break
                (score, tree, fvector), edge, vecj = heapq.heappop(cand)

                if tree not in node.kset:
                    templist.append ((score, tree, fvector))
                    node.kset.add(tree)
                else:
                    print >> logs, tree, "is duplicate"
                
                for i in xrange(edge.arity()):
                    newvecj = vecj[:i] + (vecj[i]+1,) + vecj[i+1:]

                    if newvecj not in edge.oldvecs:
                        newres = self.newgetres(edge, newvecj, forest.sent, weights)
                        if newres is not None:
                            edge.oldvecs.add (newvecj)
                            heapq.heappush(cand, (newres, edge, newvecj))

            #sort templist to node.klist
            node.klist = sorted(templist)
##            print >> logs, node, " klist = "
##            for i, (sc, tr, fv) in enumerate(node.klist):
##                print >> logs, "   " * 2, i+1, tr

        ''' no way to check_feats, unless you load local features'''
#         if self.check_feats:
#             for k, (sc, tr, fv) in enumerate(forest.root.klist):
#                 extra_fv = extract(tr, forest.sent, self.nlfeats, logprob=fv[0])
#                 fv.check(real_fv)
                
        return forest.root.klist[0]   #score, tree, fvector        


if __name__ == "__main__":

    ## test BU Decoder

    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog [options (-h)] [<features>]")
    optparser.add_option("-k", "", dest="k", type=int, help="k-best", metavar="K", default=1)
    optparser.add_option("-w", "", dest="weights", help="read weights from str", metavar="W", default="lm1=2 gt_prob=1")
    optparser.add_option("-f", "--print-features", dest="print_features", action="store_true", \
                         help="print all features (id=val)", default=False)
    optparser.add_option("-c", "--nocheck", dest="check_feats", action="store_false", \
                         help="do not check feats (default is to check)", default=True)
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", \
                         help="print result for each sentence", default=False)
    optparser.add_option("-d", "--debug", dest="debug", action="store_true", \
                         help="print result for each sentence", default=False)
    optparser.add_option("-b", "", dest="budecoder", action="store_true", \
                         help="print result for each sentence", default=False)
    optparser.add_option("", "--defaultnbest", dest="defaultnbest", help="default nbests", metavar="FILE", default=None)

    (opts, args) = optparser.parse_args()

    if opts.weights is not None:
        weights = get_weights(opts.weights)
    else:
        weights = Vector("lm1=2 gt_prob=1")

    extra_feats = None # prep_features(args)        

    decoder = LocalDecoder() #BUDecoder(opts.k, extra_feats, check_feats=False)
    decoder.set_feats(extra_feats)
    
    all_pp = Bleu()  # Parseval(), now BLEU
    decode_time, parseval_time = 0, 0
    sum_score = 0
    
    if opts.defaultnbest:
        defaultnbests = defaultdict(lambda : [])
        for line in open(opts.defaultnbest):
            defaultnbests[int(line.split()[0])].append(line.strip())
                      
    for i, forest in enumerate(decoder.load("-")):

        if forest is None:
            print >> logs, "forest %d is empty" % (i+1)
            if opts.defaultnbest:
                for line in defaultnbests[i][:opts.k]:
                    print line
            else:
                print

        else:

            score, tree, fvector, pp = decoder.decode(forest, weights)
            print >> logs, "%.2lf\t%.2lf" % (score, pp.score())

            if opts.print_features:
                for k, (sc, tr, fv) in enumerate(forest.root.klist):
                    del fv[0] ## no logprob
                    print "#%d\t%s" % (k, fv)

            else:
                if opts.k:
                    forest.lazykbest(opts.k, weights=weights)
                    for k, (sc, tr, fv) in enumerate(forest.root.klist):
                        # for david's nbest to xml
                        print "%d ||| %s ||| %.4lf" % (i, tr, sc)

                else:
                    print tree
                parseval_time -= time.time()            
                all_pp += pp
                parseval_time += time.time()

                n, e = forest.size()
    ##            print >> logs, "sent #%d %s, %d words, %d nodes, %d edges, decoded in %.2lf secs" % \
    ##                  (i+1, forest.tag, len(forest), n, e, this_time)

    ##            print score, "\t", tree, "\n", fvector
                if opts.verbose:
                    print >> logs, "bleu+1=%.4lf" % pp.score()
                if opts.debug:
                    print >> logs, score, "\t", tree #, "\n", fvector

                sum_score += score            
            

    print >> logs, "overall: %d sents. decode time in %.2lf secs (avg %.2lf secs per sent)" % \
          (i+1, decoder.decode_time, decoder.decode_time / (i+1))
    print >> logs, "overall: %d sents. parseval time in %.2lf secs (avg %.2lf secs per sent)" % \
          (i+1, parseval_time, parseval_time / (i+1))
##    print >> logs, all_pp
    print >> logs, "overall bleu = %.4lf (%.2lf)" % all_pp.score_ratio() # overall bleu
    print >> logs, "overall model-cost: %.4lf" % sum_score

            
