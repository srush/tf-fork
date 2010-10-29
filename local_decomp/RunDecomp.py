# -*- coding: utf-8 -*-
import gflags as flags
import sys,os
from itertools import *
sys.path.append(os.getenv("TFOREST"))
import gflags as flags
from ngram import Ngram
from model import Model
from forest import Forest
from svector import Vector
from subgradient import SubgradientSolver
from Decode import TreeDecoder
FLAGS = flags.FLAGS

if __name__ == "__main__":

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
    
    f = Forest.load("-", is_tforest=True, lm=None)
    for i, forest in enumerate(f, 1):
      if len(forest) < 15 : continue
      solver = SubgradientSolver(TreeDecoder(forest, weights, lm, FLAGS.lm))
      solver.run()
      break
