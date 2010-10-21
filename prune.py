#!/usr/bin/env python

''' relatively-useless pruning, after Jon Graehl.
    also called posterior pruning, used by Charniak and Johnson 2005, but only on nodes.

    basically, it is an inside-outside algorithm, with (+, max) semiring.

    \beta (leaf) = 0    
    \beta (n) = max_{e \in BS(n)} \beta(e)
    \beta (e) = max_{n_i \in tails(e)} \beta(n_i) + w(e)

    (bottom-up of \beta is done in node_and_hyperedge.py: Node.bestparse())

    \alpha (root) = 0
    \merit (e) = \alpha (n) + \beta (e),   n = head(e)     ## alpha-beta
    \alpha (n_i) max= \merit (e) - \beta (n_i),  for n_i in tails(e)

    \merit (n) = \alpha (n) + \beta (n) = max_{e \in BS(n)} \alphabeta (e)
'''

import sys, time

logs = sys.stderr

from forest import Forest
from bleu import Bleu

import gflags as flags
FLAGS=flags.FLAGS

from model import Model

def inside_outside(forest, weights):
    """Compute viterbi inside-outside scores for the forest."""

 
    forest.bestparse(weights) ## inside
    
    forest.root.alpha = 0
    all_edges = [] # for sorting (outside)

    for node in forest.reverse():   ## top-down
        if not hasattr(node, "alpha"):
            node.unreachable = True
        else:
            node.merit = node.alpha + node.beta

            for edge in node.edges:
                edge.merit = node.alpha + edge.beta

                all_edges.append((edge.merit, edge)) # for sorting (outside)
                
                for sub in edge.subs:
                    score = edge.merit - sub.beta
                    if not hasattr(sub, "alpha") or score < sub.alpha:
                        sub.alpha = score

    return all_edges
    
def prune(forest, weights, gap=None, ratio=None):
    ''' Known issue: not 100% correct w.r.t. floating point errors.'''

    def check_subs(edge, threshold):
        ''' check if every tail falls in the beam. '''
        for sub in edge.subs:
            if not hasattr(sub, "survived"):#hasattr(sub, "unreachable") or sub.merit > threshold:
                return False
        return True

    start_time = time.time()

    all_edges = inside_outside(forest, weights)
    oldsize = forest.size()

    threshold = forest.root.merit + (gap if gap is not None else 1e+9)
    if ratio is not None:
        all_edges.sort()
        allowed = int(len(forest.sent) * ratio) # allowed size
        threshold = min(threshold, all_edges[min(allowed, len(all_edges)-1)][0])
    
    newnodes = {}
    neworder = []

    kleinedges = 0
    for node in forest: # N.B.: in Forest.__iter__: nodeorder
        iden = node.iden
        if not hasattr(node, "unreachable") and node.merit <= threshold:  ## node pruning
            node.edges = [e for e in node.edges if (e.merit <= threshold and check_subs(e, threshold))]
            if node.edges == []: ## N.B.: isolated node: unreachable from bottom-up!
                print >> logs, "WARNING: isolated node found!"
            else:
                node.survived = True
                newnodes[iden] = node
                neworder.append(node)                

        else:
            kleinedges += len(node.edges)
            del node

    del forest.nodes
    del forest.nodeorder
    
    forest.nodes = newnodes
    forest.nodeorder = neworder

    forest.rehash() ## important update for various statistics
    
    newsize = forest.size()
    
    print >> logs, "%s (len: %d), gap %s, %4d nodes, %5d edges remained. prune ratio = %.1f%%, %.1f%% (%.1f%%) edges/len: %.1f" \
          % (forest.tag, len(forest.sent),
             str(gap), newsize[0], newsize[1], 
             (oldsize[0] - newsize[0])*100.0 / oldsize[0], (oldsize[1] - newsize[1])*100.0 / oldsize[1],
             kleinedges*100.0/oldsize[1],
             newsize[1]/len(forest.sent))    
    
    print >> logs, "done in %.2lf secs" % (time.time() - start_time)

    #global total_nodes, total_edges, old_nodes, old_edges
    #total_nodes += newsize[0]
    #total_edges += newsize[1]
    #old_nodes += oldsize[0]
    #old_edges += oldsize[1]


if __name__ == "__main__":

    flags.DEFINE_float("prob", None, "score threshold", short_name="p")
    flags.DEFINE_float("ratio", None, "ratio of |hyperedges|/|sent|", short_name="r")
    flags.DEFINE_boolean("oracle", False, "compute oracle after pruning")
    flags.DEFINE_boolean("out", True, "output pruned forest (to stdout)")
    flags.DEFINE_string("suffix", None, "suffix for dumping (1.<suffix>)", short_name="s")
    flags.DEFINE_integer("startid", 1, "start id for dumping")

    from ngram import Ngram # defines --lm and --order    

    argv = FLAGS(sys.argv)

    if FLAGS.prob is None and FLAGS.ratio is None:
        print >> logs, "Error: must specify pruning threshold by -p or ratio by -r" + str(FLAGS)
        sys.exit(1)

    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram() # if FLAGS.lm is None then returns None
    if lm:
        weights["lm1"] = weights["lm"] * FLAGS.lmratio
    
    onebestscores = 0
    onebestbleus = Bleu()
    myscores = 0
    myoraclebleus = Bleu()    
    
    total_nodes = total_edges = old_nodes = old_edges = 0
    
    for i, forest in enumerate(Forest.load("-", lm=lm), 1):
        if forest is None:
            print
            continue
        
        prune(forest, weights, FLAGS.prob, FLAGS.ratio)

        score, hyp, fv = forest.root.bestres
        
        forest.bleu.rescore(hyp)
        onebestscores += score
        onebestbleus += forest.bleu.copy()

        if FLAGS.oracle: #new
            bleu, hyp, fv, edgelist = forest.compute_oracle(weights, 0, 1, store_oracle=True)
            ##print >> logs, forest.root.oracle_edge
            bleu = forest.bleu.rescore(hyp)
            mscore = weights.dot(fv)
            print  >> logs, "moracle\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf" % \
                  (mscore, forest.bleu.fscore(), forest.bleu.ratio()) #, hyp) # don't output oracle
            
            myoraclebleus += forest.bleu.copy()
            myscores += mscore

        if FLAGS.out:
            if FLAGS.suffix is not None:
                forest.dump(open("%d.%s" % (i+FLAGS.startid, FLAGS.suffix), "wt"))
            else:
                forest.dump()

        if i % 10 == 0:
            if old_edges <> 0:
                print >> logs, "%d forests pruned, avg new size: %.1lf %.1lf (survival ratio: %4.1lf%% %4.1lf%%)" % \
                            (i, total_nodes/i, total_edges/i, \
                             total_nodes*100./old_nodes, total_edges*100./old_edges)
                
    print >> logs,  "overall 1-best oracle bleu = %s  score = %.4lf" \
          % (onebestbleus.score_ratio_str(), onebestscores/i)
    
    if FLAGS.oracle:
        print >> logs,  "overall my     oracle bleu = %s  score = %.4lf" \
              % (myoraclebleus.score_ratio_str(), myscores/i)


