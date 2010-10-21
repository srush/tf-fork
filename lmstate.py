#!/usr/bin/env python

import sys
from node_and_hyperedge import Hyperedge, Node
from svector import Vector
from rule import Rule

from forest import Forest

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_boolean("dp", True, "dynamic programming")    
flags.DEFINE_boolean("complete", False, "use complete step")    

class DottedRule(object):

    __slots__ = "edge", "dot", "_hash"

    def __init__(self, edge, dot=0):
        self.edge = edge
        self.dot = dot
        self.rehash()

    def rehash(self):
        self._hash = self.edge._hash + self.dot

    def tree_size(self):
        return self.edge.rule.tree_size() # number of non-variable nodes in the lhs tree

    def advance(self):
        '''advance the dot by one position (in-place!)'''
        self.dot += 1
        self.rehash()

    def advanced(self):
        '''advance the dot by one position (new!)'''
        return DottedRule(self.edge, self.dot+1)

    def next_symbol(self):
        try:
            return self.edge.lmlhsstr[self.dot] # Node or str
        except:
            print self.edge.lmlhsstr, self.dot
            assert False

    def end_of_rule(self):
        return self.dot == len(self.edge.lhsstr)

    def __eq__(self, other):
        # TODO: only compare those after dot
        return self.edge == other.edge and self.dot == other.dot

    def __str__(self):
        return self.edge.dotted_str(self.dot)

    def __hash__(self):
        return self._hash
    
class LMState(object):

    ''' stack is a list of dotted rules(hyperedges, dot_position) '''

    # backptrs = [((prev_state, ...), extra_fv, extra_words)]
    __slots__ = "stack", "_trans", "score", "step", "_hash", "backptrs", "_lmstr"

    weights = None
    lm = None
    dp = False
    lmcache = {}
    cachehits = 0
    cachemiss = 0

    @staticmethod
    def init(lm, weights, dp=None):
        LMState.lm = lm
        LMState.weights = weights
        if dp is None:
            dp = FLAGS.dp
        LMState.dp = dp

    @staticmethod
    def start_state(root):
        ''' None -> <s>^{g-1} . TOP </s>^{g-1} '''

##        LMState.cache = {}

        lmstr = LMState.lm.raw_startsyms()
        lhsstr = lmstr + [root] + LMState.lm.raw_stopsyms()
        
        edge = Hyperedge(None, [root], Vector(), lhsstr)
        edge.lmlhsstr = LMState.lm.startsyms() + [root] + LMState.lm.stopsyms()
        edge.rule = Rule.parse("ROOT(TOP) -> x0 ### ")
        sc = root.bestres[0] if FLAGS.futurecost else 0
        return LMState(None, [DottedRule(edge, dot=len(lmstr))], LMState.lm.startsyms(),
                       step=0, score=sc) # future cost

    def __init__(self, prev_state, stack, trans, step=0, score=0, extra_fv=Vector()):
        self.stack = stack
        self._trans = trans
        self._lmstr = self._trans[-LMState.lm.order+1:]
        self.step = step
        self.score = score

        # maintain delta_fv and cumulative score; but not cumul fv and delta score
        self.backptrs = [((prev_state,), extra_fv, [])] # no extra_words yet
        
        self.scan()
        

    def predict(self):
        if not self.end_of_rule():
            next_symbol = self.next_symbol()
            if type(next_symbol) is Node:
                base = self.score - (next_symbol.bestedge.beta if FLAGS.futurecost else 0)
                for edge in next_symbol.edges:
                    # N.B.: copy trans
##                    score = self.score + edge.fvector.dot(LMState.weights),
                    if FLAGS.futurecost:
                        future = sum([sub.bestedge.beta for sub in edge.subs])
                    else:
                        future = 0
                    score = base + future + edge.fvector.dot(LMState.weights)

                    yield LMState(self,
                                  self.stack + [DottedRule(edge)], 
                                  self._trans[:], 
                                  self.step + edge.rule.tree_size(),
                                  score,
                                  extra_fv=Vector()+edge.fvector) # N.B.: copy! + is faster

    def lmstr(self):
        # TODO: cache real lmstr
##        return self._trans[-LMState.lm.order+1:] if LMState.dp else self._trans
        return self._lmstr

    def scan(self):

        while not self.is_final():
            # scan
            if not self.end_of_rule():                
                symbol = self.next_symbol()
                if type(symbol) is int:
                    # TODO: cache lm index
                    this = symbol # LMState.lm.word2index(symbol)
                    self.stack[-1].advance() # dot ++
                    #TODO fix ngram
                    lmscore = LMState.lm.ngram.wordprob(this, self._lmstr)

                    self.score += lmscore * LMState.weights.lm_weight
                    _, extra_fv, extra_words = self.backptrs[0]
                    extra_fv["lm"] += lmscore
                    extra_words += [this,]
                    self._trans += [this,]
                    self._lmstr = self._trans[-LMState.lm.order+1:]
                else:
                    break
            else:
#                self.step += 0 #self.stack[-1].tree_size()
                self.stack = self.stack[:-2] + [self.stack[-2].advanced()]

        self.rehash()

    next_symbol = lambda self: self.stack[-1].next_symbol()
    end_of_rule = lambda self: self.stack[-1].end_of_rule()

    def complete(self):
        if self.end_of_rule():
            # N.B.: copy trans
            yield LMState(self,
                          self.stack[:-2] + [self.stack[-2].advanced()], 
                          self._trans[:],
                          self.step + self.stack[-1].tree_size(),
                          self.score) # no additional cost/fv

    def rehash(self):
        self._hash = hash(tuple(self.stack) + tuple(self._lmstr))# + (self.score, self.step))

    def __eq__(self, other):
        ## calls DottedRule.__eq__()
        return self.stack == other.stack and \
               self._lmstr == other._lmstr

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def is_final(self):
        ''' a complete translation'''
        # TOP' -> <s> TOP </s> . (dot at the end)
        return len(self.stack) == 1 and self.stack[0].end_of_rule()

    def trans(self, external=True):
        '''recover translation from lmstr'''
        if external:
            return LMState.lm.ppqstr(self._trans[LMState.lm.order-1 : -LMState.lm.order+1])
        else:
            return LMState.lm.ppqstr(self._trans[LMState.lm.order-1 : ])
           
    def __str__(self):
        return "LMState step=%d, score=%.2lf, trans=\"%s\", stack=[%s], lm=%.2f" % \
               (self.step, self.score, self.trans(external=False), ", ".join("(%s)" % x for x in self.stack),
                self.backptrs[0][1]["lm"])

    def __hash__(self):
        return self._hash

    def get_fvector(self):
        '''recursively reconstruct'''

        (prev_state,), extra_fv, _ = self.backptrs[0]
        if prev_state is None:
            return extra_fv
        return prev_state.get_fvector() + extra_fv

    def merge_with(self, old):
        self.backptrs += old.backptrs

    def _toforest(self, lmforest, sent, cache, level=0):
        if self in cache:
            return cache[self]

        this_node = Node("", "X [0-0]", 1, Vector(), sent) # no node fv, temporary iden=""
        is_root = level == 0 # don't include final </s> </s>
        
        for prev_states, extra_fv, extra_words  in self.backptrs:
            prev_nodes = [p._toforest(lmforest, sent, cache, level+1) for p in prev_states if p is not None]

            if is_root:
                extra_words = extra_words[:-LMState.lm.order+1]
                
            edge = Hyperedge(this_node, prev_nodes, extra_fv,
                             prev_nodes + LMState.lm.ppqstr(extra_words).split())

            edge.rule = Rule("a(\"a\")", "b", "")
            edge.rule.ruleid = 1

            this_node.add_edge(edge)
        
        cache[self] = this_node
        this_node.iden = str(len(cache)) # post-order id
        lmforest.add_node(this_node)
        if is_root:
            lmforest.root = this_node
        return this_node
        
    def toforest(self, forest):
        '''generate a forest object (for kbest)'''

        cache = {}
        lmforest = Forest(forest.sent, forest.cased_sent, is_tforest=True, tag=forest.tag)
        lmforest.refs = forest.refs

        self._toforest(lmforest, forest.sent, cache)

        return lmforest


class TaroState(LMState):

    ''' stack is a list of dotted rules(hyperedges, dot_position) '''

    # backptrs = [((prev_state, ...), extra_fv, extra_words)]

    @staticmethod
    def start_state(root):
        ''' None -> <s>^{g-1} . TOP </s>^{g-1} '''

##        LMState.cache = {}

        lmstr = LMState.lm.raw_startsyms()
        
        sc = root.bestres[0] if FLAGS.futurecost else 0
        
        return TaroState(None,
                         stack=tuple(LMState.lm.stopsyms()) + (root,),
                         trans=LMState.lm.startsyms(),
                         step=0, score=sc) # future cost
    
    def __init__(self, prev_state, stack, trans, step=0, score=0, extra_fv=Vector()):
        self.stack = stack
        self._trans = trans
        self._lmstr = self._trans[-LMState.lm.order+1:]
        self.step = step
        self.score = score

        # maintain delta_fv and cumulative score; but not cumul fv and delta score
        self.backptrs = [((prev_state,), extra_fv, [])] # no extra_words yet
        
        self.scan()        

    def predict(self):
        
        next_symbol = self.stack[-1]
        
        if type(next_symbol) is Node:
            base = self.score - (next_symbol.bestedge.beta if FLAGS.futurecost else 0)
            for edge in next_symbol.edges:
                # N.B.: copy trans
##                    score = self.score + edge.fvector.dot(LMState.weights),
                if FLAGS.futurecost:
                    future = sum([sub.bestedge.beta for sub in edge.subs])
                else:
                    future = 0

                score = base + future + edge.fvector.dot(LMState.weights)

                yield TaroState(self,
                                self.stack[:-1] + tuple(reversed(edge.lmlhsstr)),
                                self._trans[:], 
                                self.step + edge.rule.tree_size(),
                                score,
                                extra_fv=Vector()+edge.fvector) # N.B.: copy! + is faster
            
    def scan(self):
        '''actually, pop'''

        while len(self.stack) > 0:
            # scan
            symbol = self.stack[-1]
            if type(symbol) is int:
                # TODO: cache lm index
                this = symbol # LMState.lm.word2index(symbol)

                lmscore = LMState.lm.ngram.wordprob(this, self._lmstr)
                
                self.score += lmscore * LMState.weights.lm_weight
                _, extra_fv, extra_words = self.backptrs[0]
                extra_fv["lm"] += lmscore
                extra_words += [this,]
                self._trans += [this,]
                self._lmstr = self._trans[-LMState.lm.order+1:]

                self.stack = self.stack[:-1] # pop
            else:
                break

        self.rehash()

    def is_final(self):
        ''' a complete translation'''
        return len(self.stack) == 0

    def rehash(self):
        self._hash = hash(self.stack + tuple(self._lmstr))
