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
from pygraphviz import *
import gc, cPickle, time

def graph_lattice(graph):

  G=AGraph(strict=False,directed=True)
  G.graph_attr['rankdir']='LR'
      #G.charset = 'ascii'
  for n in graph:
    #print "%s -> "%n
    
    G.add_node(str(n))
    node = G.get_node(str(n))
    node.attr["color"] = n.color()
    node.attr["label"] = n.label()
    
    for n in graph:
      for e in n.edges:
        #print "\t %s"%e
        G.add_edge(str(n), str(e))
        
  G.draw("/tmp/graph.ps", prog="dot")
        

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
    
def find_trigrams(graph):
  #nodes = {}
  nodes_trigram = {} 
  forward_nodes = {}
  gc.disable()

  for n in graph:
    if n.lex:
      nodes_trigram[n.id] = set()      
    forward_nodes[n.id] = set()

  for i in range(50):
    print "Iteration", i
    change = False
    for n in graph:
      for n2 in n.edges:
        if n2.lex <> None:
          forward_nodes[n.id].add(n2.id)
        else: 
          forward_nodes[n.id] = forward_nodes[n.id] | forward_nodes[n2.id]
      
  print "enumerate"
  for n in graph:
    if not n.lex: continue
    for firstn in forward_nodes[n.id]:
      for secondn in forward_nodes[firstn]:
        assert n.lex <> None
        assert graph.nodes[firstn].lex <> None  
        assert graph.nodes[secondn].lex <> None
        ngram = ((n.lex,0), (graph.nodes[firstn].lex,0), (graph.nodes[secondn].lex,0) )
        nodes_trigram[n.id].add(ngram)
        

  return (forward_nodes,nodes_trigram)

def find_back_trigrams(graph):
  # for each node, find the lex nodes "before" it 
  
  #nodes = {}
  back_nodes = {}
  gc.disable()

  for n in graph:
    
    back_nodes[n.id] = set()

  for i in range(50):
    print "Iteration", i
    change = False
    for n in graph:
      for n2 in n.back_edges:
        if n2.lex <> None:
          back_nodes[n.id].add(n2.id)
        else: 
          back_nodes[n.id] = back_nodes[n.id] | back_nodes[n2.id]
      
  print "enumerate"
        
  return back_nodes

def get_trigrams(graph, back_nodes):
  for n in graph:
    if not n.lex: continue
    for firstn in back_nodes[n.id]:
      for secondn in back_nodes[firstn]:
        assert n.lex <> None
        assert graph.nodes[firstn].lex <> None  
        assert graph.nodes[secondn].lex <> None
        ngram = ((graph.nodes[secondn].lex,0), (graph.nodes[firstn].lex,0), (n.lex,0) )
        nodes_trigram[n.id].add(ngram)
  return nodes_trigram  

def graph_node(graph):
  
  for n in graph:
    if n.lex:
      node = n
      break

  nodes = [node]
  seen_nodes = set()
  G=AGraph(strict=False,directed=True)
  G.graph_attr['rankdir']='LR'

  gc.disable()
  i = 0
  while nodes:
    i+=1
    if i ==400: break
    
    
    n = nodes.pop(0)
    if n.id in seen_nodes: continue
    seen_nodes.add(n.id)
    
    G.add_node(str(n))
    gnode = G.get_node(str(n))
    gnode.attr["color"] = n.color()
    
    gnode.attr["label"] = str(n.label())
    for n2 in n.edges:
      nodes.append(n2)
      G.add_edge(str(n), str(n2))
        
  G.draw("/tmp/graph.ps", prog="dot")

      

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

    flags.DEFINE_boolean("graph", False, "")
    flags.DEFINE_boolean("enumerate", False, "")
    flags.DEFINE_boolean("graph_node", False, "")
    flags.DEFINE_boolean("size", False, "")
    
    
    argv = FLAGS(sys.argv)

    assert not (FLAGS.size and FLAGS.graph and FLAGS.enumerate and FLAGS.graph_node)  

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
              words.add(n)       
      print "Words", len(words)

      print "Worst case", len(words) * len(words) * len(words)

      graph = NodeExtractor().extract(forest)
      
      if FLAGS.graph:
        graph_lattice(graph)

      elif FLAGS.graph_node:
        graph_node(graph)

      elif FLAGS.size:
        back_nodes  = find_back_trigrams(graph)
        
        total = 0 
        for n in graph:
          total += len(back_nodes[n.id])
        print "total is", total
        print "worst case is", len(words) * graph.size()
  
        tritotal = 0 
        for n in graph:
          for n2 in back_nodes[n.id]:
            tritotal += len(back_nodes[n2])
        print "total is", tritotal
        print "worst case is", len(words) * len(words) * graph.size()
  
        print "Starting"
        sys.stdout.flush()
        for i in range(300 * 300 * 300):
          lm.word_prob_bystr("a", ["he", "the"])


      elif FLAGS.enumerate: 
        
        nodes_tri= get_trigrams(graph, find_back_trigrams(graph))
        totaltri = 0
        for n in graph:
          if n.lex:
            totaltri += len(nodes_tri[n.id])
          
        print "total", totaltri

        for n in graph:
          if n.lex:
            print strip_lex(n.lex)
            newls = []
            for l2 in nodes_tri[n.id]:
              if len(l2) == 3:
                assert(l2[2][0] == n.lex)
                assert l2[0][0] <> None
                assert l2[1][0] <> None
                assert l2[2][0] <> None
                
                s = lm.word_prob_bystr(strip_lex(l2[2][0]), [strip_lex(l2[0][0]), strip_lex(l2[1][0])])
                newls.append((s, strip_lex(l2[0][0]), strip_lex(l2[1][0])))

            newls.sort()    
            newls.reverse()
            for s, w1, w2 in newls:
              print "\t %s  \t%s\t%.3f"%(  w1, w2, s)
        print len(graph.first.edges)
      break
