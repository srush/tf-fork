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


def write_files(forest, graph, lm):
  words = set()
  #for n in forest:
  #    for edge in n.edges:
  #        for i, n in enumerate(edge.rule.rhs):
  #            if is_lex(n):
  #                words.add(strip_lex(n))       
                    
  for n in graph:
    if n.lex:
      words.add(strip_lex(n.lex))

  word_map = {}
  word_file = open("/tmp/words", 'w')
  print  "PROBS", lm.word_prob("according to economic \" taiwan \" , cross @-@ strait trade 2 zero nine billion us dollars last year .")
  for i, w in enumerate(words):
      print >>word_file, i, lm.word2index(w)
      word_map[lm.word2index(w)] = i
      for j, w2 in enumerate(words):
        print w, w2, lm.word_prob_bystr(w2, w)
  

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
    
    
