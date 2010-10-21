#!/usr/bin/env python

''' Two classes that constitute a hypergraph (forest): Node and Hyperedge
    On top of these, there is a separate Forest class in forest.py which collects the nodes,
    and deals with the loading and dumping of forests.

    implementation details:
    1. node has a local score "node_score" and hyperedge "edge_score".
    2. and they both have "beta" \'s.

    this design is quite different from the original, where only edge-prob is present.
    
'''

import sys, os, re
import math
import copy

#sys.path.append(os.environ["NEWCODE"])

#import mycode
import heapq

logs = sys.stderr

from tree import Tree

from svector import Vector
from utility import symbol, desymbol

print_duplicates = False

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_boolean("bp", True, "use BP rules")

class Node(Tree):
    ''' Node is based on Tree so that it inherits various functions like binned_len and is_terminal. '''

    def copy(self):
        return copy.deepcopy(self)
    
    def __init__(self, iden, labelspan, size, fvector, sent):
        # NP [0-3]
        self.iden = iden
        
        label, span = labelspan.split()
        self.span = tuple(map(int, span[1:-1].split("-")))
        
        if label[-1] == "*":
            label = label[:-1]
            self._spurious = True
        else:
            self._spurious = False
            
        self.label = "TOP" if label == "S1" else label
        self.label = symbol(self.label)
        self.edges = []
        
        #new features
        self.frags = []
        #self.tfedges = []

        #new feature: subtree str created for bp rules, NP(NN 'ch') -> lhs(bp) ### feats 
        self.subtree = ''
        
        ## N.B.: parse forest node can be termllinal
        word = sent[self.span[0]] if (size == 0) else None

        ## now in MT forest, nodes are always non-final. hyperedges can be final (terminal).

        ## in tree.py
        self.prepare_stuff(label, word)

        self.fvector = fvector

        self._root = False

        self._bin_len = None

        # surface string
        self.surface = '%s' % ''.join(sent[self.span[0]:self.span[1]])

        self._hash = hash(self.iden)

    def __hash__(self):
        return self._hash

    def psubtree(self):
        if self.subtree != '':
            return self.subtree
        else:
            if self.is_terminal():
                self.subtree = '%s("%s")' % (self.label, self.word)
                return self.subtree
            else:
                self.subtree = '%s(%s)' % \
                               (self.label, ' '.join(sub.psubtree() for sub in self.edges[0].subs))
                return self.subtree
                             
                
    def prepare_kbest(self):
        self.klist = []
        self.kset = set()
        self.fixed = False
        # if self.is_terminal(): --- WHY??/

        if self.is_terminal():
            self.klist = [self.bestres]
            self.kset.add(tuple(self.besttree))
            self.fixed = True
               
##        self.bestedge = None  ## N.B.: WHY??
        self.cand = None

    def mapped_span(self, mapping):
        return (mapping[self.span[0]], mapping[self.span[1]]) 
    
    def labelspan(self, separator=":", include_id=True, space=" "):
        ss = "%s%s " % (self.iden, separator) if include_id else ""
        lbspn = "%s%s[%d-%d]" % (self.label + ("*" if self.is_spurious() else ""),
                                 space,
                                 self.span[0], self.span[1])
        return ss + lbspn

    __str__ = labelspan
    __repr__ = __str__
    
    def is_spurious(self):
        return self._spurious

    def sp_terminal(self):
        return self.is_spurious() and self.edges[0].subs[0].is_terminal()

    def add_edge(self, hyperedge):
        self.edges.append(hyperedge)
        hyperedge.node = self ## important backpointer!

    def number_edges(self, counter, newedgeorder):
        retcount = counter
        for edge in self.edges:
            edge.position_id = retcount
            newedgeorder.append(edge)
            retcount += 1 
        return retcount

    def assemble(self, subtrees):
        '''this is nice. to be used by k-best tree generation.'''
        ## t = Tree(self.label, self.span, subs=subtrees, sym=False) if not self._spurious else subtrees[0]

        ## now done in hyperedge, not here in node

        assert False, "shoudn't be called here, see Hyperedge."
            
        assert t is not None, (self.label, self.span, subtrees, self._spurious)
#          if self._root:
#             ## notice that, roots are spurious! so...
#             t.set_root(True)
        return t

    def this_tree(self):
        ## very careful: sym=False! do not symbolize again
        return Tree(self.label, self.span, wrd=self.word, sym=False)

    def sumparse(self, weights=Vector("gt_prob=1")):
        # memoize
        if hasattr(self,"betasum"):
            return self.betasum

        self.node_score = (self.fvector.dot(weights))

        score = 0
        for i, edge in enumerate(self.edges):
            #if FLAGS.bp or not edge.rule.is_bp():
            edge.edge_score = (edge.fvector.dot(weights))  # TODO
            
            edge.betasum = 1.0 
            for sub in edge.subs:
                sub_beta = sub.sumparse(weights)
                edge.betasum *= sub_beta
                
            edge.betasum *= edge.edge_score
            score += edge.betasum
            
        self.betasum = score + self.node_score
        return self.betasum



#     def sumparse(self, weights=Vector("gt_prob=1")):
#         # memoize
#         if hasattr(self,"betasum"):
#             return (self.insidepaths, self.betasum)

#         self.node_score = self.fvector.dot(weights)

#         insidepaths = 0 
#         score = 0

#         # not sure the name of this semiring,
#         # but it computes + ((a + b),(a' + b')) and * ((a * b), (b * a' + a * b'))
#         # gives the sum of all derivations 

#         for i, edge in enumerate(self.edges):
#             #if FLAGS.bp or not edge.rule.is_bp():
#             edge.edge_score = edge.fvector.dot(weights)  # TODO
            
#             edge.betasum = 0.0 
#             edge.insidepaths = 1
#             for sub in edge.subs:
#                 sub_insidepaths, sub_beta = sub.sumparse(weights)
#                 edge.betasum = edge.insidepaths * sub_beta + sub_insidepaths * edge.betasum 
#                 edge.insidepaths *= sub_insidepaths 


#             edge.betasum += edge.edge_score * edge.insidepaths
#             score += edge.betasum
#             insidepaths += edge.insidepaths


#         self.insidepaths = insidepaths 
#         self.betasum = score + self.node_score
#         return (self.insidepaths, self.betasum)

    def bestparse(self, weights, use_min, dep=0):
        '''now returns a triple (score, tree, fvector) '''
        
        if self.bestres is not None:
            return self.bestres

        self.node_score = self.fvector.dot(weights) 
        if self._terminal:
            self.beta = self.node_score
            
            self.besttree = self.this_tree()
            self.bestres = (self.node_score, self.besttree, self.fvector.__copy__())  ## caution copy TODO

        else:

            self.bestedge = None
            for edge in self.edges:
                if FLAGS.bp or not edge.rule.is_bp():
                    
                    ## weights are attached to the forest, shared by all nodes and hyperedges
                    score = edge.edge_score = edge.fvector.dot(weights)  # TODO
                    fvector = edge.fvector.__copy__() ## N.B.! copy! TODO
                    subtrees = []
                    for sub in edge.subs:
                        sc, tr, fv = sub.bestparse(weights, use_min, dep+1)
                        score += sc
                        fvector += fv
                        subtrees.append(tr)

                    tree = edge.assemble(subtrees)

                    edge.beta = score
                    if self.bestedge is None or (score < self.bestedge.beta if use_min else score > self.bestedge.beta):
    ##                    print >> logs, self, edge
                        self.bestedge = edge
                        self.besttree = tree
                        best_fvector = fvector

            self.beta = self.bestedge.beta + self.node_score
            best_fvector += self.fvector ## nodefvector

            self.bestres = (self.beta, self.besttree, best_fvector)

        return self.bestres

    def print_derivation(self, dep=0):

        if not self.is_terminal():
            print "  " * dep, self.labelspan()
            for sub in self.bestedge.subs:
                sub.print_derivation(dep+1)
            print

    def getcandidates(self, dep=0):
        self.cand = []
        for edge in self.edges:
            vecone = edge.vecone()
            edge.oldvecs = set([vecone])
            res = edge.getres(vecone, dep)
            assert res, "bad at candidates"
            self.cand.append( (res, edge, vecone) )
            
        heapq.heapify (self.cand)
        
    def lazykbest(self, k, dep=0):

        now = len(self.klist)
##        print >> logs, self, k, now
        if self.fixed or now >= k:
            return

        if self.cand is None:
            self.getcandidates(dep)
            self.last_edge_vecj = None
            
        if self.last_edge_vecj is not None:
            edge, vecj = self.last_edge_vecj
            edge.lazynext(vecj, self.cand, dep+1)
            
        while now < k:
            if self.cand == []:
                self.fixed = True
                return 
            
            (score, tree, fvector), edge, vecj = heapq.heappop(self.cand)
            if tuple(tree) not in self.kset:
                ## assemble dynamically
                self.klist.append ((score, tree, fvector))
                self.kset.add(tuple(tree))
                now += 1
            else:
                if print_duplicates:
                    print >> logs, "*** duplicate %s: \"%s\", @%d(k=%d)" % (self, tree, now, k)  #labespan

            if now < k:  ## don't do extra work if you are done!
                edge.lazynext(vecj, self.cand, dep+1)
                self.last_edge_vecj = None
            else:
                self.last_edge_vecj = (edge, vecj)

    def get_oracle_edgelist(self):
        assert hasattr(self, "oracle_edge"), self
        edge = self.oracle_edge

        es = [edge]
        for sub in edge.subs:
            es += sub.get_oracle_edgelist()

        return es

    def compute_oracle(self, weights, fbleu, flen, fwlen, model_weight=0, bleu_weight=1, memo=None):

        if memo is None:
            memo = {}        

        if self.iden in memo:
            return memo[self.iden]

        bleu = fbleu.copy()
        ratio = self.span_width() / float(flen) 
        bleu.special_reflen = fbleu.single_reflen() * ratio # proportional reflen
            
        best_score = float("-inf")        
        
        best_fv = None

        wlen = ratio * fwlen
        for edge in self.edges:
            fv = edge.fvector.__copy__() + self.fvector.__copy__() #N.B.:don't forget node feats!
            edges = [edge]
            hyps = []
            for sub in edge.subs:
                sub_s, sub_h, sub_fv, sub_es = sub.compute_oracle(weights, fbleu, flen, fwlen, model_weight, bleu_weight, memo)
                edges += sub_es
                hyps.append(sub_h)
                fv += sub_fv

            hyp = edge.assemble(hyps) ## TODO: use LM states instead!
            bleu_score = bleu.rescore(hyp) ## TODO: effective ref len!
            model_score = weights.dot(fv)
            #print wlen, ratio, flen, bleu.rescore(hyp), hyp
            ## interpolate with 1-best weights
            score = bleu_score * wlen * bleu_weight - model_score * model_weight    # relative!
            
            if score > best_score or \
                         model_weight == 0 and math.fabs(score - best_score) < 1e-4 and \
                         (best_fv is None or model_score < best_model_score):

                best_score = score
                best_bleu_score = bleu_score
                best_model_score = model_score
                best_edges = edges
                best_hyp = hyp
                best_fv = fv
        
        memo[self.iden] = (best_bleu_score, best_hyp, best_fv, best_edges)

        return memo[self.iden]
            

    
def substitute(varstr, subtrees):
    ''' now returns a str!'''

    s = []
    varid = 0
    for w in varstr:
        if type(w) is str:
            #w = desymbol(w) ## N.B.: unquote here!
            if w != "":  ## @UNKNOWN@ => "", do not include it
                s.append(w)
        else:
            if subtrees[varid] != "":        # single @UNKNOWN@        
                s.append(subtrees[varid])
            varid += 1

    return " ".join(s)
            

class Hyperedge(object):

    def unary(self):
        return not self.head.is_root() and len(self.subs) == 1

    def unary_cycle(self):
        return self.unary() and self.subs[0].label == self.head.label
    
    def __str__(self):
        return "%-17s  ->  %s " % (self.head, "  ".join([str(x) for x in self.subs]))

    def shorter(self):
        ''' shorter form str: NP [3-5] -> DT [3-4]   NN [4-5]'''
        return "%s  ->  %s " % (self.head.labelspan(include_id=False), \
                                "  ".join([x.labelspan(include_id=False) \
                                           for x in self.subs]))

    def dotted_str(self, dot):
        ''' NP [3-5] -> DT [3-4] . NN [4-5]'''
        rhs = [(x.labelspan(include_id=False, space="") \
                if type(x) is Node else x) for x in self.lhsstr]
        rhs.insert(dot, ".")

        return "%s -> %s" % (self.head, " ".join(rhs))

    def shortest(self):
        ''' shortest form str: NP -> DT NN '''
        return "%s  ->  %s " % (self.head.label, "  ".join([str(x.label) for x in self.subs]))            
                             
    __repr__ = __str__

    def __init__(self, head, tails, fvector, lhsstr):
        self.head = head
        self.subs = tails
        self.fvector = fvector

        # lhsstr is a list of either variables (type "Node") or strings
        # like ["thank", node_5, "very", "much"]
        self.lhsstr = lhsstr
        #self.rhsstr = rhsstr
        self._hash = hash((head, ) + tuple(lhsstr))

    def arity(self):
        return len(self.subs)

    def vecone(self):
        return (0,) * self.arity()

    def compatible(self, tree, care_POS=False):
        if self.arity() == tree.arity():
            
            for sub, tsub in zip(self.subs, tree.subs):
                if not sub.compatible(tsub, care_POS):
                    return False
            return True

    def assemble(self, subtrees):
        return substitute(self.lhsstr, subtrees)

    def getres(self, vecj, dep=0):
        score = self.edge_score 
        fvector = self.fvector + self.head.fvector
        subtrees = []
        for i, sub in enumerate(self.subs):

            if vecj[i] >= len(sub.klist) and not sub.fixed:
                sub.lazykbest(vecj[i]+1, dep+1)
            if vecj[i] >= len(sub.klist):
                return None
            
            sc, tr, fv = sub.klist[vecj[i]]
            subtrees.append(tr)
            score += sc
            fvector += fv

        return (score, self.assemble(subtrees), fvector)

    def lazynext(self, vecj, cand, dep=0):
        for i in xrange(self.arity()):
            ## vecj' = vecj + b^i (just change the i^th dimension
            newvecj = vecj[:i] + (vecj[i]+1,) + vecj[i+1:]

            if newvecj not in self.oldvecs:
                newres = self.getres(newvecj, dep)
                if newres is not None:
                    self.oldvecs.add (newvecj)
                    heapq.heappush(cand, (newres, self, newvecj))

    @staticmethod
    def _deriv2tree(edgelist, i=0):
        '''convert a derivation (a list of edges) to a tree, using assemble
           like Tree.parse, returns (pos, tree) pair
        '''
        edge = edgelist[i]
        node = edge.head
        subs = []
        for sub in edge.subs:
            if not sub.is_terminal():
                i, subtree = Hyperedge._deriv2tree(edgelist, i+1)
            else:
                subtree = sub.this_tree()
            subs.append(subtree)

        return i, edge.assemble(subs)

    @staticmethod
    def deriv2tree(edgelist):
        _, tree = Hyperedge._deriv2tree(edgelist)
        ## list => string
        return tree
    
    @staticmethod
    def deriv2fvector(edgelist):
        '''be careful -- not only edge fvectors, but also node fvectors, including terminals'''
        
        fv = Vector()
        for edge in edgelist:
            fv += edge.fvector + edge.head.fvector
            for sub in edge.subs:
                if sub.is_terminal():
                    fv += sub.fvector
        return fv

    def __hash__(self):
        return self._hash
