#!/usr/bin/env python
""" Forest Oracle """

import sys, time
import math
import utility
logs = sys.stderr
from bleu import Bleu
from cubepruning import FeatureScorer, CubePruning

sys.path.append('..')

from cyksearch import CYKDecoder
class BleuScorer(object):
  def __init__(self, weights, bleu_weight = 1.0, model_weight=0.0):
    self.weights = weights
    self.bleu_weight = bleu_weight
    self.model_weight = model_weight
    
  def from_edge(self, edge):
    fvector = edge.fvector + edge.head.fvector
    return (None, -self.weights.dot(fvector), fvector)
    #return None
  
  def times(self, one, two):
    #if two[1] == None:
    return (two[0], one[1] + two[1], one[2] + two[2])
    #else:
    #  return (None, two[1] + one[1], one[2] + two[2])
    #(sc1, fv1) = one
    #(sc2, fv2) = two
    #return (sc1 + sc2, fv1 + fv2) 
    

  def __cmp__(self, other):
    return cmp(other, self)

  def add(self, one, two):
    return one


#         bleu = fbleu.copy()
#         ratio = self.span_width() / float(flen) 
#         bleu.special_reflen = fbleu.single_reflen() * ratio # proportional reflen
            
#         best_score = float("-inf")        
        
#         best_fv = None

#         wlen = ratio * fwlen
#         for edge in self.edges:
#             fv = edge.fvector.__copy__() + self.fvector.__copy__() #N.B.:don't forget node feats!
#             edges = [edge]
#             hyps = []
#             for sub in edge.subs:
#                 sub_s, sub_h, sub_fv, sub_es = sub.compute_oracle(weights, fbleu, flen, fwlen, model_weight, bleu_weight, memo)
#                 edges += sub_es
#                 hyps.append(sub_h)
#                 fv += sub_fv

#             hyp = edge.assemble(hyps) ## TODO: use LM states instead!
#             bleu_score = bleu.rescore(hyp) ## TODO: effective ref len!
#             model_score = weights.dot(fv)

#             ## interpolate with 1-best weights
#             score = bleu_score * wlen * bleu_weight - model_score * model_weight    # relative!
            
#             if score > best_score or \
#                          model_weight == 0 and math.fabs(score - best_score) < 1e-4 and \
#                          (best_fv is None or model_score < best_model_score):

#                 best_score = score
#                 best_bleu_score = bleu_score
#                 best_model_score = model_score
#                 best_edges = edges
#                 best_hyp = hyp
#                 best_fv = fv


def oracle_extracter(forest, weights, false_decoder, k, ratio, extract=1):
  "reimplementation of forest.compute_oracle using cube pruning to get oracle forest"

  flen = len(forest)
  fbleu = forest.bleu
  def non_local_scorer(cedge, ders):
    bleu = fbleu.copy()

    node = cedge.head
    
    ratio = node.span_width() / float(flen) 
    bleu.special_reflen = fbleu.single_reflen() * ratio # proportional reflen
    wlen = ratio * flen
    
    hyp = cedge.assemble(ders)
    #print wlen, ratio, flen, bleu.rescore(hyp), hyp,ders
    bleu_score = bleu.rescore(hyp) #- (float(len(hyp.split()))* 1e-5)
    fv = Vector()
    if false_decoder:
      (lmsc, alltrans, sig) = false_decoder.deltLMScore(cedge.lhsstr, ders)
      fv["lm"] = lmsc    
    return ((bleu_score * wlen, -weights.dot(fv), fv), hyp, hyp)  

  decoder = CubePruning(BleuScorer(weights, 1.0, 0.0), non_local_scorer, k, ratio, find_min=False)

  start = time.time()
  best = decoder.run(forest.root)
  end = time.time()
  print >> logs, "Cube Bleu %s"%((end - start))

  start = time.time()
  dec_forest = decoder.extract_kbest_forest(forest, extract)
  end = time.time()
  print >> logs, "Extracting Forest %s"%((end - start))
  
  return dec_forest, best
  



# ---------------------MAIN------------------
import gflags as flags
FLAGS=flags.FLAGS

from svector import Vector
def main():
  from ngram import Ngram
  from model import Model
  from forest import Forest
  
  flags.DEFINE_integer("beam", 100, "beam size", short_name="b")
  flags.DEFINE_integer("debuglevel", 0, "debug level")
  flags.DEFINE_boolean("mert", True, "output mert-friendly info (<hyp><cost>)")
  flags.DEFINE_boolean("cube", True, "using cube pruning to speedup")
  flags.DEFINE_integer("kbest", 1, "kbest output", short_name="k")
  flags.DEFINE_integer("ratio", 3, "the maximum items (pop from PQ): ratio*b", short_name="r")
  

  argv = FLAGS(sys.argv)
  [outfile] = argv[1:]
  weights = Model.cmdline_model()
  lm = Ngram.cmdline_ngram()
  

  false_decoder = CYKDecoder(weights, lm)
  out = utility.getfile(outfile, 1)
  old_bleu = Bleu()
  new_bleu = Bleu()
  
  for i, forest in enumerate(Forest.load("-", is_tforest=True, lm=lm), 1):
    
    oracle_forest, oracle_item = oracle_extracter(forest, weights, false_decoder, 100, 2, extract=100)
    print >>sys.stderr, "processed sent %s " % i
    oracle_forest.dump(out)
    bleu, hyp, fv, edgelist = forest.compute_oracle(weights, 0.0, 1)

    forest.bleu.rescore(hyp)
    old_bleu += forest.bleu
    forest.bleu.rescore(oracle_item[0].full_derivation)
    new_bleu += forest.bleu

    bad_bleu, _, _, _ = oracle_forest.compute_oracle(weights, 0.0, -1)
    #for i in range(min(len(oracle_item), 5)):
     # print >>sys.stderr, "Oracle Trans: %s %s %s" %(oracle_item[i].full_derivation, oracle_item[i].score, str(oracle_item[i].score[2]))
     # print >>sys.stderr, "Oracle BLEU Score: %s"% (forest.bleu.rescore(oracle_item[i].full_derivation))
    print >>sys.stderr, "Oracle BLEU Score: %s"% (forest.bleu.rescore(oracle_item[0].full_derivation))
    print >>sys.stderr, "Worst new Oracle BLEU Score: %s"% (bad_bleu)
    print >>sys.stderr, "Old Oracle BLEU Score: %s"% (bleu)
    
    print >>sys.stderr, "Running Oracle BLEU Score: %s"% (new_bleu.compute_score())
    print >>sys.stderr, "Running Old Oracle BLEU Score: %s"% (old_bleu.compute_score())

    #assert a[0], b[0].score[0]
    #assert a[1], b[0].score[1]

    # print forest.bleu.rescore(b[1].full_derivation)
#     print forest.bleu.rescore(a[1])
#     print a
#     print b[0]


if __name__ =="__main__": main()
