
import gflags as flags
from util import *
import sys
from itertools import *
sys.path.append('..')
import gflags as flags
from ngram import Ngram
from model import Model
from forest import Forest
from svector import Vector
from lattice_extractor import *



def query_trigram(node):
  result = []
  queue = [([node], frozenset([node.id]), 1)]
  while queue:

    (ns, hist, lnum) = queue.pop(0)


    if lnum == 3:
      result.append(ns)
    else:
      n = ns[-1]

      if isinstance(n, LexNode):
        newlnum = lnum + 1
      else:
        newlnum = lnum
      
      for n2 in n.edges:
        if n2 not in hist:
          queue.append((ns + [n2], hist | frozenset([n2.id]), newlnum))
  return results
    


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
    #lm = Ngram.cmdline_ngram()
    

    f = Forest.load("-", is_tforest=True, lm=None)
    for i, forest in enumerate(f, 1):
      words = set()
      for n in forest:
        for edge in n.edges:
          for i, n in enumerate(edge.rule.rhs):
            if is_lex(n):
              words.add(n)       
      print len(words)
      
      graph = NodeExtractor().extract(forest)

      for n in graph:
        print "%s -> "%n
        #for e in n.edges:
          #print "\t %s"%e
        print query_trigram(n)
      print len(graph.first.edges)