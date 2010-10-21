#!/usr/bin/env python
""" general version of cube pruning (Alg. 2) to apply whenever, partially for my learning """

from __future__ import division

import sys
import time
from collections import defaultdict
from foresthelpers import general_prune
logs = sys.stderr
import heapq
from itertools import *


class FeatureScorer(object):
  def __init__(self, weights):
    self.weights = weights
    
  def from_edge(self, edge):
    fvector = edge.fvector + edge.head.fvector
    return (self.weights.dot(fvector), fvector)

  def times(self, one, two):
    (sc1, fv1) = one
    (sc2, fv2) = two
    return (sc1 + sc2, fv1 + fv2) 

  def add(self, one, two):
    return one
  

class Item(object):
  def __str__(self):
    return str((self.score, self.full_derivation, self.sig))
  def __init__(self, score, full_derivation, last_edge, sig, scorer, find_min = True):
    """
    @param score - some semiring like structure
    @param full_derivation - opaque derivation
    @param last_edge - the last edge we took to get to this derivation
    @param sig - some structure with well-defined Eq  
    """
    self.score = score
    self.full_derivation = full_derivation
    self.last_edges = [last_edge]
    self.sig = sig
    self.find_min = find_min
    self.scorer = scorer
  def add(self, other):
    # don't need to do anything (assume sorted)
    self.last_edges.extend(other.last_edges)
    self.score = self.scorer.add(self.score, other.score)
  def __cmp__(self, other):
    "Hack: need heapq to find large elements when find_min is false"
    if self.find_min:
      return cmp(self.score, other.score)
    else :
      return cmp(other.score, self.score)

class CubePruning(object):
    """
    Implements a general version of cube-pruning algorithm 2.

    Goal is to be close to the paper, be stateless as possible, and to abstract away things like features and language model  
    """
    def __init__(self, scorer, non_local, k, ratio, find_min = True):
      self.non_local_feature_function = non_local
      self.scorer = scorer
      self.hypothesis_cache = {}
      self.k = k
      self.ratio = ratio
      self.find_min = find_min
    

    def extract_kbest_forest(self, forest, kbest = 1):
      "extract the forest created by the kbest final parses"
      
      marked_edges = set()
      done_nodes = set([forest.root])
      
      edge_stack = sum([item.last_edges for item in self.hypothesis_cache[forest.root][0:kbest]], [])
      #print [str(edge.rule) for edge in edge_stack]
      #print "Marked edges"
      while edge_stack:
        #print edge_stack
        edge, vecj = edge_stack.pop()
        marked_edges.add(edge.position_id)
        for sub, j in izip(edge.subs, vecj):
          #if sub in done_nodes: continue
          #done_nodes.add(sub)
          item = self.hypothesis_cache[sub][j]
          #print item.full_derivation
          edge_stack.extend(item.last_edges)
      

          
      
      def node_pruning(node):
        return False
        
      def edge_pruning(edge):
        return edge.position_id not in marked_edges
          
      return general_prune(forest, node_pruning, edge_pruning)


    def extract_pruned_forest(self, forest, extract = 1):
      "extract the forest created by cube pruning"
      def node_pruning(node):
        if self.hypothesis_cache.has_key(node):
          return False
        return True
        
      def edge_pruning(edge):
        #print "need edge %s %s" %(edge,edge.position_id)
        node = edge.head
        hypvec = self.hypothesis_cache[node]
        for i, item in enumerate(hypvec):
          #print "Last edge is %s %s" % (item.last_edge,item.last_edge.position_id)
          if edge.position_id in [e.position_id for e in item.last_edges] :
            return False
          if i == extract: break
        return True
          
      return general_prune(forest, node_pruning, edge_pruning)

      
    def run(self, cur_node):
        "compute the k-'best' list for cur_node" 
        for hedge in cur_node.edges:
            for sub in hedge.subs:
                if not self.hypothesis_cache.has_key(sub):                    
                    self.run(sub)

        # create cube
        cands = self.init_cube(cur_node)
        heapq.heapify(cands)
        
        # gen kbest
        self.hypothesis_cache[cur_node] = self.kbest(cands) 
        #print cur_node
        #print map(str,self.hypothesis_cache[cur_node])

        return self.hypothesis_cache[cur_node]
        
    def init_cube(self, cur_node):
        
        cands = []
        for cedge in cur_node.edges:
            cedge.oldvecs = set()

            # start with (0,...0)
            newvecj = (0,) * cedge.arity()
            cedge.oldvecs.add(newvecj)

            # add the starting (0,..,0) hypothesis to the heap
            newhyp = self.gethyp(cedge, newvecj)
            cands.append((newhyp, cedge, newvecj))
            
        return cands
        
    def kbest(self, cands):
        """
        Algorithm 2, kbest 
        """

        # list of best hypvectors (buf)
        hypvec = []
        
        # number of hypotheses found 
        cur_kbest = 0

        # keep tracks of sigs in buffer (don't count multiples twice, since they will be recombined)
        sigs = set()

        # overfill the buffer since we assume there will be some reordering
        buf_limit = self.ratio * self.k

        while cur_kbest < self.k and \
                  not (cands == [] or \
                       len(hypvec) >= buf_limit):

            (chyp, cedge, cvecj) = heapq.heappop(cands)

            #TODO: duplicate management
            
            if chyp.sig not in sigs:
                sigs.add(chyp.sig)
                cur_kbest += 1

            # add hypothesis to buffer
            hypvec.append(chyp)

            # expand next hypotheses
            self.next(cedge, cvecj, cands)
 

        # RECOMBINATION (shrink buf to actual k-best list)
        
        # sort and combine hypevec
        hypvec.sort()

        keylist = {}
        
        newhypvec = []

        for item in hypvec:
            if not keylist.has_key(item.sig):
                keylist[item.sig] = len(newhypvec)
                newhypvec.append(item)

                if len(newhypvec) >= self.k:
                  break

            else:
                pos = keylist[item.sig]
                
                # semiring plus
                newhypvec[pos].add(item)

                
        return newhypvec
    
    def next(self, cedge, cvecj, cands):
        """
        @param cedge - the edge that we just took a candidate from
        @param cvecj - the current position on the cedge cube
        @param cands - current candidate list 
        """
        # for each dimension of the cube
        for i in xrange(cedge.arity()):
            ## vecj' = vecj + b^i (just change the i^th dimension
            newvecj = cvecj[:i] + (cvecj[i]+1,) + cvecj[i+1:]
            if newvecj not in cedge.oldvecs:
                newhyp = self.gethyp(cedge, newvecj)
                if newhyp is not None:
                    # add j'th dimension to the cube
                    cedge.oldvecs.add(newvecj)
                    heapq.heappush(cands, (newhyp, cedge, newvecj))
                        
    def gethyp(self, cedge, vecj):
        """
        Return the score and signature of the element obtained from combining the
        vecj-best parses along cedge. Also, apply non-local feature functions (LM)
        """
        score = self.scorer.from_edge(cedge)

        subders = []

        # grab the jth best hypothesis at each node of the hyperedge
        for i, sub in enumerate(cedge.subs):
          
            if vecj[i] >= len(self.hypothesis_cache[sub]):
                return None


            item = self.hypothesis_cache[sub][vecj[i]]
            subders.append(item.full_derivation)
            score = self.scorer.times(score, item.score)

        # Get the non-local feature and signature information
        (non_local_score, full_derivation, sig) = self.non_local_feature_function(cedge, subders)

        score = self.scorer.times(score, non_local_score)
        return Item(score, full_derivation, (cedge, vecj), sig, self.scorer, self.find_min)
    
