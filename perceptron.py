#!/usr/bin/env python

import sys
import math
import time
import copy

from forest import Forest, get_weights
from decoder import Decoder, LocalDecoder
from svector import Vector
##from features import *
##from local_feat import *
from bleu import Bleu

logs = sys.stderr

import mymonitor

compare = cmp   ## for now, just use string-cmp; TODO: use parseval-equality

update_mode = "perc"  # use "perc" for perceptron, "MIRA" for MIRA.
use_avg = True
use_loss = False

##def offline_extract(tree, sent, score):
##    fv = extract(tree, sent, all_feats)
##    fv[0] = score ## put back logprob
##    return fv

def G(d):
    return min(d, 1) ## bound by 1, so that update isn't too radical

def MIRA(deltafv, weights, loss):

    if update_mode == "perc":
        return 1
    elif update_mode == "MIRA":
##         print "original deltafv =\n", deltafv
##         print "current weights =\n", weights
        dotproduct = weights * deltafv
        norm2 = deltafv.norm2()
##        print >> logs, "\n     w*deltafv=%.6lf, norm2=%.6lf" % (dotproduct, norm2)
##        print "dotproduct =", dotproduct, "norm_2 =", norm2
        return G(dotproduct / norm2)
    else:
        assert False, "update_mode %s not implemented" % update_mode
    
def one_example(forest, weights, indent=" " * 5):
    ''' perceptron on one example. '''

    if opts.fear: #fear
        bestscore, besttree, bestfvector, parseval = decoder.fear(forest, weights)        
    else:    #1-best
        bestscore, besttree, bestfvector, parseval = decoder.decode(forest, weights)

    if opts.check_fvector:
        realfvector = extract(besttree, forest.sent, all_feats, logprob=bestscore)
        assert bestfvector == realfvector, "diff in feature vectors\n%s" % (realfvector - bestfvector)  

    if isinstance(decoder, LocalDecoder):
        ### N.B. the parseval returned by decoder is against gold_tree,
        ### while here we need the distance from oracle tree
        bleuscore = forest.bleu.score(besttree)
        loss = forest.oracle_bleu_score - bleuscore # Parseval(besttree, forest.oracle_tree).fscore()
    else:
        ## NBest: if same as oracle, then 0, otherwise parseval against gold
        ## N.B.: "is" is not OK for lists of strings
        if besttree == forest.oracle_tree:
            loss = 1
        else:
            loss = parseval.fscore()
        
    if loss > 1e-6: ## 0.xxxxyy
        deltafv = forest.oracle_fvector - bestfvector

        if update_mode == "MIRA":            
            update_rate = MIRA(deltafv, weights, loss)        
        else:
            if opts.use_loss:
                update_rate = loss
                if opts.num_brackets is not None:
                    update_rate *= forest.oracle_size_ratio

            else:
                update_rate = 1

        
        print >> logs, indent, "bleu+1 = %.4lf, |delta| = %d, update_rate = %.5lf" % \
              (bleuscore, len(deltafv), update_rate)
        if opts.debug_update:
            print >> logs, indent, "oracle =", forest.oracle_fvector
            print >> logs, indent, "oracle =", forest.oracle_tree
            print >> logs, indent, "besttr =", bestfvector
            print >> logs, indent, "besttr =", besttree
            print >> logs, indent, "best==oracle: ", besttree == forest.oracle_tree
            print >> logs, indent, "update delta fvector-----\n", deltafv
        updated = 1

        if math.fabs(update_rate - 1) > 1e-6:
            deltafv *= update_rate            

    else:
        updated = 0
        deltafv = None ## no update
        print >> logs, indent, "PASS! :)"

    return updated, parseval, deltafv

def evaluate_all(weights, tests):
    
    parseval = Bleu() # Parseval()
    for forest in tests:
        bestscore, besttree, bestfv, pp = decoder.decode(forest, weights)
        #print "score=%.4lf\tbleu+1=%.4lf" % (bestscore, pp.score())
        parseval += pp #Parseval(besttree, forest.goldtree)

    #print parseval.score()
    return parseval

class AVG_Weights(Vector):
    ''' self.last_update is a mapping from feature f to a pair (last_it, last_i),
        denoting the last iteration and last example that this feature was updated, respectively.        
    '''

    def __init__(self, d={}):
        Vector.__init__(self, d)
        self.last_update = {}
        self.N = 0 ## number of examples (per iter). will be reset at the end of the first iteration

    def add(self, update, current, it, i):
        ''' update is the update vector that will be added to current.
            here I pay back all my debt (in current) if they reoccur in update,
            and record new debt from update. '''
        
        for f in update:
            if f in current:
                last_it, last_i = self.last_update.get(f, (0, 0))
                ## pay the last debt on this feature
                self[f] += current[f] * ((it - last_it) * self.N + (i - last_i))

            ## record new debt
            self.last_update[f] = (it, i)

    def final(self, current, it, N):
        '''return a copy of the currently averaged weights, but do not edit in place'''
        
        self.N = N
        ## pay all my debts up to this point, as if there is an extra round,
        ## where every feature is updated, by 0
        self.add(current, current, it, N)
        return self                        ## do not need to copy()

def output_weights(weights):

    s = "%s" % weights
    d = filter(lambda (f,v): math.fabs(float(v))>1e-4, [x.split("=") for x in s.split()])
    return " ".join(["%s=%s" % (f,v) for (f,v) in sorted(d)])
    

if __name__ == "__main__":

    try:
         import psyco
         psyco.full()
        print >> logs, "psyco imported."
    except:
        pass

    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <forests> | %prog [options (-h)] <lfeats> <nlfeats>")
    optparser.add_option("-i", "--iter", dest="iterations", type=int, \
                         help="number of iterations (a.k.a. epochs)", metavar="T", default=1)
    optparser.add_option("-N", "--nbest", dest="N", type=int, \
                         help="nbest", metavar="N", default=50)
    optparser.add_option("-k", "--kbest", dest="k", type=int, \
                         help="kbest", metavar="K", default=1)
    optparser.add_option("-s", "--shuffle", dest="shuffle", action="store_true", \
                         help="randomize the order of examples", default=False)
    optparser.add_option("-l", "--local", dest="local", action="store_true", default=False, \
                         help="extract local features (default is already stored in .local)") 
    optparser.add_option("-M", "--MIRA", dest="MIRA", action="store_true", default=False, \
                         help="1-best MIRA update")
    optparser.add_option("-L", "--Loss", dest="use_loss", action="store_true", default=False, \
                         help="update sensitive to Loss")
    optparser.add_option("-B", "--Brackets", dest="num_brackets", type=float, default=20.0, \
                         help="update sensitive to Loss and number of brackets over LEN", metavar="LEN")
    optparser.add_option("-m", "--mode", dest="mode", type=int, default=1, \
                         help="mode (0: nbest, 1: local, 2: forest-BU)")
    optparser.add_option("", "--test", dest="testfile", metavar="FILE", default=None, \
                         help="(single) test file")
    optparser.add_option("", "--dev", dest="devfile", metavar="FILE", default=None, \
                         help="(single) dev file")
    optparser.add_option("", "--train", dest="trainfile", metavar="FILE", default=None, \
                         help="(single) dev file")
    optparser.add_option("", "--noavg", dest="use_avg", action="store_false", default=True, \
                         help="no averaging of weights")
    optparser.add_option("-D", "--debug-update", dest="debug_update", action="store_true", default=False, \
                         help="print weight update (\delta \W) vectors") 
    optparser.add_option("-A", "--adaptive", dest="adaptive", type=int, default=None, metavar="base", \
                         help="adaptive base for different span widths") 
    optparser.add_option("", "--slow-avg", dest="slow_avg", action="store_true", default=False, \
                         help="naive avg update (and sanity check)") 
    optparser.add_option("", "--check-fvector", dest="check_fvector", action="store_true", default=False, \
                         help="sanity checking fvector from decoding and from offline extraction")
    optparser.add_option("-w", "", dest="weightsfile", help="read weights from", metavar="FILE", default=None)
    optparser.add_option("-o", "--weightsoutprefix", dest="weightsoutprefix", help="output weights file prefix", metavar="FILE", default=None)
    optparser.add_option("", "--hope", dest="hope", type=float, help="hope weight on modelcost", metavar="r", default=1e-10)
    optparser.add_option("", "--shrink", dest="shrink_weights", action="store_true", default=False, \
                         help="shrink model weights to norm 1 after each iteration") 
    optparser.add_option("", "--fear", dest="fear", action="store_true", default=False, \
                         help="(oracle|hope) - fear (instead of 1best)") 
    optparser.add_option("", "--switch", dest="switch", action="store_true", default=False, \
                         help="switch to 1best after overfitting") 

    (opts, args) = optparser.parse_args()

    if opts.mode is None or opts.trainfile is None:
        optparser.error("must specify train (--train <FILE>). ")

    if opts.num_brackets is not None:
        Decoder.MAX_NUM_BRACKETS = opts.num_brackets

        
    if opts.MIRA:
        update_mode = "MIRA"
    use_avg = opts.use_avg
    use_loss = opts.use_loss

    print >> logs, "update mode = %s, with avg = %s, with Loss = %s" % (update_mode, use_avg, use_loss)

##    decoder = [NBestDecoder(opts.N), \
##               LocalDecoder(), \
##               BUDecoder(opts.k, check_feats=False, adaptive_base=opts.adaptive)]\
##               [opts.mode]

    decoder = LocalDecoder(opts.hope)
    print >> logs, "decoder = %s" % decoder

    ### must read forest first! otherwise slow!
#     forests = []
#     for forest in decoder.load("-"):
#         forests.append(forest)

    if opts.weightsfile is not None:
        weights = get_weights(opts.weightsfile) # see forest.py
    else:
        weights = Vector("lm1=2 gt_prob=1")  ## initial vector
        
    initial_weights = weights.__copy__()

    extra_feats = None #prep_features(args)

    decoder.set_feats(extra_feats)
    all_feats = extra_feats

    if opts.trainfile == "-":
        trainforests = []
        for forest in decoder.load(opts.trainfile):
            decoder.do_oracle(forest, weights)
            trainforests.append(forest)
            
        print >> logs, "pre-loaded %d train forests, load %.2lf, oracle %.2lf" % \
              (len(trainforests), decoder.load_time(), decoder.oracle_time)
        preloaded = True
    else:
        preloaded = False
##        trainforests = decoder.load(opts.trainfile)

    sum_weights = Vector()
    total_weights = AVG_Weights()

    oldtime = time.time()
    if opts.devfile is not None:
        devforests = decoder.load(opts.devfile)
        devscore = evaluate_all(weights, devforests).score() #.fscore()
    else:
        devscore = -1

    print >> logs, "at the end of iteration 0, errors_on_train = -, ", \
                "train = -, dev = %.2lf, " % (devscore*100), \
                "|avgW| = %d" % len(weights)

    best_devscore = -1

    print >> logs, "starting perceptron at", time.ctime()
    for it in xrange(opts.iterations):

        print >> logs, "iteration %d" % (it+1), "= = " * 20
        print >> logs, "hope weight on modelcost = %lf" % opts.hope

        iterstart = time.time()

        if opts.shuffle:
            ## TODO: randomize
            pass

        parseval = Bleu()
        num_updates = 0

        avgtime = 0
        decoder.reset()
        
        if not preloaded:
            trainforests = decoder.load(opts.trainfile)
            

        for i, forest in enumerate(trainforests):

            decoder.do_oracle(forest, weights)
            
            print >> logs, "  iteration %d, example %d" % (it+1, i+1), "-" * 5, "oracle = %.4lf" % forest.oracle_bleu_score, 
            updated, pp, deltafv = one_example(forest, weights)
            parseval += pp
            num_updates += updated


            avgtime -= time.time()

            if deltafv is not None:
                ## lock some features; see also forest.py
                #del deltafv["outm1n"]
                #del deltafv["psm1n"]
                
                ## fast update here
                total_weights.add(deltafv, weights, it, i)
                ## update here
                weights -= deltafv ## because we always minimize!
                ## hack!
                if opts.shrink_weights or (update_mode == "MIRA" and i % 50 == 0):
                    # re-shrinking back to norm 1 (every 50 sentences):
                    rate = 1/math.sqrt(weights.squarednorm()) #used to be: 1/weights.norm()
                    weights *= rate
                    print >> logs,"   weights re-shrinked by", rate

                print >> logs, "   weights 2-norm", weights.squarednorm() #, output_weights(weights)
                
            ## legacy code, to check
            if opts.slow_avg:
                sum_weights += weights
                
            avgtime += time.time()
            

        total_weights.final(weights, it, i+1)

        if opts.slow_avg:
            assert total_weights.check(sum_weights)
            avg_weights = sum_weights / ((it+1) * (i+1))

        avged_weights = total_weights / ((it+1) * (i+1))

        
        itertime = time.time() - iterstart
        
        print >> logs, "iteration %d time %.2lf (%.2lf per sent)" % (it+1, itertime, itertime/(i+1)),
        print >> logs, "decode %.2lf, averaging %.2lf" % (decoder.decode_time, avgtime),
        if not preloaded:
            print >> logs, "load %.2lf, oracle %.2lf, extract %.2lf" % \
                  (decoder.load_time(), decoder.oracle_time, decoder.extract_time)
        else:
            print >> logs

        trainscore = parseval.score() #.fscore()

        ## re-load
        if opts.devfile is not None:
            devforests = decoder.load(opts.devfile)
            devscore = evaluate_all(avged_weights, devforests).score() #.fscore()
        else:
            devscore = -1
            
        if opts.testfile is not None:
            testforests = decoder.load(opts.testfile)
            
        testscore = evaluate_all(avged_weights, testforests).score() \
                                if opts.testfile is not None else 0

        scores = "train = %.4lf, dev = %.2lf, test = %.2lf" % (trainscore, devscore*100, testscore*100)

        print >> logs, "at the end of iteration %d, errors_on_train = %d, " % (it+1, num_updates),
        print >> logs, scores,
        print >> logs, "|avgW| = %d" % len(avged_weights)
        
        print >> logs, "iteration %d eval time %.2lf" % (it+1, time.time() - (iterstart + itertime))

##        print "weights at iteration %d, " % (it+1) + scores
        if devscore > best_devscore:
            best_devscore = devscore
            best_iteration = it + 1  # 1-based
            best_weightlen = len(avged_weights)

            # output weights to a new file
            weightsoutfile = open("%s.at%d.bleu%.2f" % (opts.weightsoutprefix, it+1, devscore*100), "wt")
            print >> weightsoutfile, output_weights(avged_weights)
            weightsoutfile.flush()
            
        else:
            ## overfitting already
            if opts.switch:
                print >> logs, "switching to oracle - 1best."
                opts.fear = False # switch to (oracle) - 1best
                
        sys.stdout.flush()

        mymonitor.gc_collect()

    print >> logs, "best dev score = %.4lf @ iteration %d, |avgW| = %d" % (best_devscore, best_iteration, best_weightlen)
    newtime = time.time()    
    print >> logs, "finishing perceptron at", time.ctime(), " total time = %.2lf (%.2lf per iter)" % \
          (newtime - oldtime, (newtime-oldtime)/opts.iterations)
