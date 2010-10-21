#!/usr/bin/env python
""" general version of cube pruning (Alg. 2) to apply whenever """

from __future__ import division

import sys
import gflags as flags


sys.path.append('..')
from cubepruning import FeatureScorer, CubePruning
from cyksearch import CYKDecoder
logs = sys.stderr
FLAGS=flags.FLAGS

if __name__ =="__main__": main()
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

  weights = Model.cmdline_model()
  lm = Ngram.cmdline_ngram()
  
  false_decoder = CYKDecoder(weights, lm)
  
  def non_local_scorer(cedge, ders):
    (lmsc, alltrans, sig) = false_decoder.deltLMScore(cedge.lhsstr, ders)
    fv = Vector()
    fv["lm"] = lmsc
    return ((weights.dot(fv), fv), alltrans, sig)
  cube_prune = CubePruning(FeatureScorer(weights), non_local_scorer, FLAGS.k, FLAGS.ratio)

  for i, forest in enumerate(Forest.load("-", is_tforest=True, lm=lm), 1):
    a = false_decoder.beam_search(forest, b = FLAGS.beam)
    b = cube_prune.run(forest.root)

    assert a[0], b[0].score[0]
    assert a[1], b[0].score[1]
    print a
    print b[0]


