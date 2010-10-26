# -*- coding: utf-8 -*-
import gflags as flags
from util import *
import sys,os
from itertools import *
sys.path.append(os.getenv("TFOREST"))
import gflags as flags
from ngram import Ngram
from model import Model
from forest import Forest
from svector import Vector
from lattice_extractor import *

import gc, cPickle, time


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
      if len(forest) < 20 : continue
      words = set()
      for n in forest:
        for edge in n.edges:
          for i, n in enumerate(edge.rule.rhs):
            if is_lex(n):
              words.add(strip_lex(n))       
      print "Words", len(words)

      word_map = {}
      word_file = open("/tmp/words", 'w')
      for i, w in enumerate(words):
        print >>word_file, i, lm.word2index(w)
        word_map[lm.word2index(w)] = i
        
      graph = NodeExtractor().extract(forest)
      graph.filter((lambda n: isinstance(n, NonTermNode)))  
      graph_file = open("/tmp/graph", 'w')
      for n in graph:
        if n.lex:
          type = 0
          val = word_map[lm.word2index(strip_lex(n.lex))]
        else:
          type = 1
          val = 0 # TODO: FIXME

        if n.id == graph.size() -1: final = 1
        else: final = 0
        print >>graph_file, n.id, final, len(n.edges), type, val,
        for e in n.edges:
          print >>graph_file, e.id,
        print >>graph_file
      
      break
