#!/usr/bin/env python

import sys, math
import itertools, heapq, collections, random
import re, xml.sax.saxutils
import sym, rule, cost, svector, log

import oracle
import sgml

import collections

logs = sys.stderr
print_states_warning = False # don't warn inconsistent states

from utility import words_to_chars
from bleu import Bleu

if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

def quoteattr(s):
    return '"%s"' % s.replace('\\','\\\\').replace('"', '\\"')

def quotefeature(s):
    return xml.sax.saxutils.escape(s, { ',' : '&comma;' ,
                                        ':' : '&colon;' , 
                                        '=' : '&equals;' ,
                                        ',' : '&comma;' ,
                                        '(' : '&lrb;' ,
                                        ')' : '&rrb;' })

def strstates(models, states):
    return ", ".join(m.strstate(s) for m,s in itertools.izip(models, states))

class Derivation(object):
    ## lhuang: TODO: should be based on dict, not object
    def __init__(self, goal):
        '''lhuang: a mapping from node to hyperedge. '''
        self.ded = {}
        self.goal = goal

    def english(self, item=None):
        if item is None:
            item = self.goal
        ded = self.ded[id(item)]

        antes = [self.english(ant) for ant in ded.ants]
        r = ded.rule
        if r is not None:
            # lhuang: substitute, nice (lambda)
            e = r.e.subst((), antes)
        elif len(antes) == 1: # this is used by the Hiero goal item
            e = antes[0]

        return e
        
    def vector(self, item=None):
        if item is None:
            item = self.goal
        ded = self.ded[id(item)]
        v = svector.Vector(ded.dcost)
        for ant in ded.ants:
            v += self.vector(ant)

        return v

    def select(self, item, ded):
        self.ded[id(item)] = ded

    def _str_helper(self, item, accum):
        ded = self.ded[id(item)]
        if ded.rule:
            x = ded.rule.lhs
        else:
            x = sym.fromtag("-")
        if len(ded.ants) > 0:
            accum.extend(["(", sym.totag(x)])
            for ant in ded.ants:
                accum.append(" ")
                self._str_helper(ant, accum)
            accum.append(")")
        else:
            accum.append(sym.totag(x))

    def __str__(self):
        accum = []
        self._str_helper(self.goal, accum)
        return "".join(accum)


class NBestInfo(object):
    """Information about an Item that is needed for n-best computation"""
    __slots__ = "nbest", "cands", "index", "english", "ecount"
    def __init__(self, item):
        self.nbest = []    # of (viterbi,ded,antranks)
        self.cands = []    # priority queue of (viterbi,ded,antranks)
        self.index = set() # of (ded,antranks)
        self.english = []
        self.ecount = collections.defaultdict(int)
        for ded in item.deds:
            zeros = (0,)*len(ded.ants)
            self.cands.append((ded.viterbi, ded, zeros))
            self.index.add((ded,zeros))
        heapq.heapify(self.cands)

class NBest(object):
    def __init__(self, goal, ambiguity_limit=None):
        self.goal = goal
        self.nbinfos = {}
        self.ambiguity_limit = ambiguity_limit

    def len_computed(self):
        return len(self.nbinfos[id(self.goal)].nbest)

    def compute_nbest(self, item, n):
        """Assumes that the 1-best has already been found
        and stored in Deduction.viterbi"""

        if id(item) not in self.nbinfos:
            self.nbinfos[id(item)] = NBestInfo(item)
        nb = self.nbinfos[id(item)]

        while len(nb.nbest) < n and len(nb.cands) > 0:
            # Get the next best and add it to the list
            (cost,ded,ranks) = heapq.heappop(nb.cands)

            if self.ambiguity_limit:
                # compute English string
                antes = []
                for ant, rank in itertools.izip (ded.ants, ranks):
                    self.compute_nbest(ant, rank+1)
                    antes.append(self.nbinfos[id(ant)].english[rank])
                if ded.rule is not None:
                    e = ded.rule.e.subst((), antes)
                elif len(antes) == 1: # this is used by the Hiero goal item
                    e = antes[0]

                # don't want more than ambiguity_limit per english
                nb.ecount[e] += 1
                if nb.ecount[e] <= self.ambiguity_limit:
                    nb.nbest.append((cost,ded,ranks))
                    nb.english.append(e)
            else:
                nb.nbest.append((cost,ded,ranks))

            # Replenish the candidate pool
            for ant_i in xrange(len(ded.ants)):
                ant, rank = ded.ants[ant_i], ranks[ant_i]

                if self.compute_nbest(ant, rank+2) >= rank+2:
                    ant_nb = self.nbinfos[id(ant)]
                    nextranks = list(ranks)
                    nextranks[ant_i] += 1
                    nextranks = tuple(nextranks)
                    if (ded, nextranks) not in nb.index:
                        nextcost = cost - ant_nb.nbest[rank][0] + ant_nb.nbest[rank+1][0]
                        heapq.heappush(nb.cands, (nextcost, ded, nextranks))
                        nb.index.add((ded,nextranks))

        return len(nb.nbest)

    def __getitem__(self, i):
        self.compute_nbest(self.goal, i+1)
        return self._getitem_helper(self.goal, i, Derivation(self.goal))

    def _getitem_helper(self, item, i, deriv):
        nb = self.nbinfos[id(item)]
        _, ded, ranks = nb.nbest[i]

        deriv.select(item, ded)

        for ant,rank in itertools.izip(ded.ants, ranks):
            self._getitem_helper(ant, rank, deriv)

        return deriv

class Item(object):
    '''In an and/or graph, this is an or node'''
    __slots__ = "x", "i", "j", "states", "deds", "viterbi", "id", "edges_str", "wi", "wj", "nodeid" # 
    ## lhuang: x is the nonterm
    def __init__(self, x, i, j, deds=None, states=None, viterbi=None):
        if type(x) is str:
            x = sym.fromstring(x)
        self.x = x
        self.i = i
        self.j = j
        self.deds = deds if deds is not None else []
        self.states = states
        self.viterbi = viterbi

    def __hash__(self):
        return hash((self.x,self.i,self.j,tuple(self.states)))

    def __cmp__(self, other):
        if other is None:
            return 1 # kind of weird
        if self.states == other.states and self.x == other.x and self.i == other.i and self.j == other.j:
            return 0
        return 1

    def __str__(self):
        if self.x is None:
            return "[Goal]"
        else:
            return "[%s,%d,%d,%s,cost=%s]" % (sym.tostring(self.x),self.i,self.j,str(self.states),self.viterbi)

    # this is used by extractor.py
    def derive(self, ants, r, dcost=0.0):
        self.deds.append(Deduction(ants, r, dcost))
        # update viterbi?

    # This actually no longer gets used
    def merge(self, item):
        self.deds.extend(item.deds)

        # best item may have changed, so update score
        if item.viterbi < self.viterbi:
            self.viterbi = item.viterbi

    # Pickling
    def __reduce__(self):
        return (Item, (sym.tostring(self.x), i, j, self.deds))

    # Postorder traversal
    def __iter__(self):
        return self.bottomup()

    def bottomup(self, visited=None):
        if visited is None:
            visited = set()
        if id(self) in visited:
            return
        visited.add(id(self))
        for ded in self.deds:
            for ant in ded.ants:
                for item in ant.bottomup(visited):
                    yield item
        yield self

    def compute_inside(self, weights, insides=None, beta=1.):
        if insides is None:
            insides = {}
        if id(self) in insides:
            return insides
        inside = cost.IMPOSSIBLE
        for ded in self.deds:
            # beta = 0 => uniform
            c = weights.dot(ded.dcost)*beta
            for ant in ded.ants:
                ant.compute_inside(weights, insides)
                c += insides[id(ant)]
            insides[id(ded)] = c
            inside = cost.add(inside, c)
        insides[id(self)] = inside
        return insides

    def compute_outside(self, weights, insides, beta=1.):
        outsides = {}
        outsides[id(self)] = 0.
        topological = list(self.bottomup())
        for item in reversed(topological):
            if id(item) not in outsides:
                # not reachable from top
                outsides[id(item)] = cost.IMPOSSIBLE
                continue
            for ded in item.deds:
                if len(ded.ants) == 0:
                    continue
                # p = Pr(ded)
                p = weights.dot(ded.dcost)*beta + outsides[id(item)]
                for ant in ded.ants:
                    p += insides[id(ant)]
                for ant in ded.ants:
                    if id(ant) not in outsides:
                        outsides[id(ant)] = p-insides[id(ant)]
                    else:
                        outsides[id(ant)] = cost.add(outsides[id(ant)], p-insides[id(ant)])
        
        if outsides is None:
            outsides = {}
        if id(self) in outsides:
            return outsides
        inside = cost.IMPOSSIBLE
        for ded in self.deds:
            # beta = 0 => uniform
            c = weights.dot(ded.dcost)*beta
            for ant in ded.ants:
                ant.compute_inside(weights, insides)
                c += insides[id(ant)]
            insides[id(ded)] = c
            inside = cost.add(inside, c)
        insides[id(self)] = inside
        return insides

    def viterbi_deriv(self, deriv=None, weights=None):
        if deriv is None:
            deriv = Derivation(self)

        viterbi_ded = min((ded.viterbi,ded) for ded in self.deds)[1]
        deriv.select(self, viterbi_ded)
        for ant in viterbi_ded.ants:
            ant.viterbi_deriv(deriv)
        return deriv

    def random_deriv(self, insides, deriv=None):
        if deriv is None:
            deriv = Derivation(self)
            
        r = random.random()
        p = 0.
        for ded in self.deds:
            p += cost.prob(insides[id(ded)]-insides[id(self)])
            if p > r:
                break
        else: # shouldn't happen
            ded = self.deds[-1]

        deriv.select(self, ded)

        for ant in ded.ants:
            ant.random_deriv(insides, deriv)

        return deriv

    def rescore(self, models, weights, memo=None, add=False, check_states=False):
        """Recompute self.viterbi and self.states according to models
        and weights. Returns the Viterbi vector, and (unlike the
        decoder) only calls weights.dot on vectors of whole
        subderivations, which is handy for overriding weights.dot.

        If add == True, append the new scores instead of replacing the old ones.
        """

        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]

        # lhuang: vviterbi means "vector Viterbi"
        vviterbi = None
        self.states = None
        for ded in self.deds:
            ded_vviterbi, states = self.rescore_deduction(ded, models, weights, memo, add=add)
            if self.states is None:
                self.states = states
            elif check_states and states != self.states:
                # don't check state at the root because we don't care
                # lhuang: LM intersection
                if print_states_warning:
                    log.write("warning: Item.rescore(): id(ded)=%s: inconsistent states %s and %s\n" % (id(ded), strstates(models, states), strstates(models, self.states)))
            if vviterbi is None or ded.viterbi < self.viterbi:
                vviterbi = ded_vviterbi
                self.viterbi = weights.dot(vviterbi)

        memo[id(self)] = vviterbi
        return vviterbi

    def rescore_deduction(self, ded, models, weights, memo, add=False):
        """Recompute ded.dcost and ded.viterbi according to models and weights."""

        vviterbi = svector.Vector()
        for ant in ded.ants:
            vviterbi += ant.rescore(models, weights, memo, add=add, check_states=True)

        if not add:
            ded.dcost = svector.Vector()
        states = []
        for m_i in xrange(len(models)):
            antstates = [ant.states[m_i] for ant in ded.ants]
            if ded.rule is not None:
                j1 = ded.ants[0].j if len(ded.ants) == 2 else None
                (state, mdcost) = models[m_i].transition(ded.rule, antstates, self.i, self.j, j1)
            elif len(antstates) == 1: # goal item
                mdcost = models[m_i].finaltransition(antstates[0])
                state = None
            states.append(state)

            ded.dcost += mdcost
        vviterbi += ded.dcost
        ded.viterbi = weights.dot(vviterbi)

        return vviterbi, states

    def reweight(self, weights, memo=None):
        """Recompute self.viterbi according to weights. Returns the
        Viterbi vector, and (unlike the decoder) only calls
        weights.dot on vectors of whole subderivations, which is handy
        for overriding weights.dot."""

        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]

        vviterbi = None
        for ded in self.deds:
            ded_vviterbi = svector.Vector()
            
            for ant in ded.ants:
                ded_vviterbi += ant.reweight(weights, memo)

            ded_vviterbi += ded.dcost
            ded.viterbi = weights.dot(ded_vviterbi)

            if vviterbi is None or ded.viterbi < self.viterbi:
                vviterbi = ded_vviterbi
                self.viterbi = ded.viterbi

        memo[id(self)] = vviterbi
        return vviterbi

    ## lhuang:    
    def adjust_spans(self, flen, fwlen, nodememo=None):
        
        if nodememo is None:
            nodememo = set()
            
        if id(self) in nodememo:
            return
        nodememo.add(id(self))
        for ded in self.deds:
            for sub in ded.ants:
                sub.adjust_spans(flen, fwlen, nodememo)

        ## swap back boundaries. TODO: replace in text
        try:
            self.i, self.j = flen - self.j, flen - self.i
            self.wi, self.wj = fwlen - self.wj, fwlen - self.wi
        except:
            print >> logs, self.i, self.j, flen, fwlen
            sys.exit(1)
        
        
    def dump(self, rules=None, sid=1, fsent="<foreign-sentence>", byline="", reflines=[]):

        nodememo = {}   # to keep track of sizes (# of nodes, # of edges)
        # forest id, foreign sentence (TODO: refs)

        fsent = fsent.split(" ")

        s = "%s\t%s\n" % (sid, " ".join(fsent)) + \
            "%d\n" % len(reflines) + \
            "".join(reflines)


        flen = len(words_to_chars(fsent, encode_back=True))        
        fwlen = len(fsent)

        reversed_fsent = list(reversed(fsent))  ## RIGHT TO LEFT
        
        if byline != "":
            self.traverse(0, 0, reversed_fsent, rules, nodememo)
            ## swap back
            self.adjust_spans(flen, fwlen)

            byline = byline.split(" ")
            byline_flen = self.i
            byline_fwlen = self.wi
            byline_f = fsent[:byline_fwlen]

            print >> logs, "clen (non-byline) = %d (%d)" % (flen, self.j - self.i)
            print >> logs, "wlen (non-byline) = %d (%d)" % (fwlen, self.wj - self.wi)            
            print >> logs, "BYLINE = " + " ".join(byline_f) + \
                  " ### %d chars, %d words" % (byline_flen, byline_fwlen)

            assert len(words_to_chars(byline_f)) == byline_flen, "@sentence %d, BYLINE Error" % opts.sentid ## check consistency

            ## new rule/edge
            ## TOP("by" "line" x0:TOP) -> "BY" "LINE" x0 ### id=-1

            byline_e = " ".join('"%s"' % w for w in byline)
            lhs = "TOP(" + byline_e + " x0:%s)" % self.x  # "TOP"
            rhs = " ".join('"%s"' % w for w in byline_f) + " x0"
            # byline rule, id=-1
            rid = -1
            rules[rid] = "%s -> %s ### id=%d" % (lhs, rhs, rid)

            ## make david-style LHS
            david_lhs = []
            for w in byline:
                david_lhs.append(sym.fromstring(w))
            david_lhs.append(sym.setindex(dummylabel, 1))
            
            ded = Deduction([self], rule.Rule(rid, rule.Phrase(david_lhs), rule.Phrase(david_lhs)),\
                            svector.Vector())
            ded.lhsstr = byline_e.split() + [self] ## N.B.: dont forget "..."
            ded.ruleid = rid
            
            # new node on top of TOP
            oldtop = self
            self = Item(self.x, 0, flen, deds=[ded])
            self.x = oldtop.x
            self.wi = 0
            self.wj = fwlen
            self.id = len(nodememo)+1
            nodememo[id(self)] = (self.id, nodememo[id(oldtop)][1]+1) #edges


            
        else:
            # establish node spans 
            self.traverse(0, 0, reversed_fsent, rules, nodememo)

            # swap i,j 
            self.adjust_spans(flen, fwlen)


        ## lhuang: the following is from hope.py
        ## be very careful about weights interpolation
        sg = sgml.Sentence(fsent)
        sg.fwords = fsent
        sg.refs = [refline.split(" ") for refline in reflines]

        if sg.refs:
            
            theoracle.input(sg, verbose=False)
            # 1-best
            self.reweight(weights)

            output(self, "1-best @ %s" % sid, onebestbleus, onebestscores)


            base_oracleweights = theoracle.make_weights(additive=True)
            # we use the in-place operations because oracleweights might be
            # a subclass of Vector

            for relative in []:#[opts.hope]:
                oracleweights = theoracle.make_weights(additive=True)
                oracleweights *= relative

                # interpolation: taking modelcost into account
                oracleweights += weights

                # compute oracle
                self.rescore(theoracle.models, oracleweights, add=True)
                # TODO: why??
                output(self, "hope%s  " % relative, hopebleus[relative], hopescores[relative])
            

        # right boundary should match sentence length (in chars)
        assert self.j == flen and self.wj == fwlen, \
               "@sentence %d, Boundary Mismatch at %s\t%s" % (opts.sentid, sid, fsent) + \
               "self.j=%d, flen=%d;  self.wj=%d, fwlen=%d" % (self.j, flen, self.wj, fwlen)        
        
        s += "%d\t%d\n" % nodememo[id(self)] + \
             self._dump(rules, deriv=self.viterbi_deriv())
        
        return s        

    def traverse(self, right_idx=0, right_widx=0, fsent=None, rules=None, nodememo=None):        
        ''' helper called by dump(); returns a string; figure out span'''

        if nodememo is None:
            nodememo = {}

        if id(self) in nodememo:
            return

        deds = [(ded.dcost.dot(weights), ded) for ded in self.deds]
        deds.sort()
        
        deds = [x for _, x in deds[:max_edges_per_node]]
        self.deds = deds # prune!

        nedges = len(deds)  # accumulating number of edges, recursively
        
        self.i = right_idx
        self.wi = right_widx

        for dedid, ded in enumerate(deds):
            try:
                rule = rules[ded.ruleid]
            except:
                print >> sys.stderr, "WARNING: rule %d not found" % ded.ruleid
                ## assuming it's a one-word UNKNOWN rule
                ## TODO: check with lattice
                unkword = fsent[self.wi]
                rule = 'UNKNOWN("@UNKNOWN@") -> "%s"' % unkword  # in reverse order
                rules[ded.ruleid] = rule
                print >> sys.stderr, "         covering " + unkword
                
                
            self.x = rule.split("(", 1)[0]  # non-terminal label

            # analyse RHS (chinese side)
            lhs, rhs = rule.split(" -> ", 1) ## -> might be a word

            # deal with lhs; convert to ded.lhsstr = ["...", "...", Item(...), "..."]
            varid = 0
            lhsstr = []
            for child in ded.rule.e:
                if sym.isvar(child):
                    lhsstr.append(ded.ants[varid])
                    varid += 1
                else:
                    lhsstr.append(quoteattr(sym.tostring(child)))

            # will be used in _dump()
            ded.lhsstr = lhsstr                
            
            vars = []
            chars_in_gap = 0
            words_in_gap = 0
            for it in reversed(rhs.split()):  ## from RIGHT to LEFT!! N.B. can't split(" ")
                if it[0] == "x":
                    #variable:
                    var = int(it[1:])
                    vars.append((var, chars_in_gap, words_in_gap))
                    chars_in_gap = 0
                    words_in_gap = 0
                else:
                    # strip off quotes "..."
                    it = it[1:-1]
                    # calculate char-length
                    if it == foreign_sentence_tag: # <foreign-sentence>:
                        # glue symbol is not counted!
                        chars_in_gap += 0
                        words_in_gap += 0
                    else:
                        # 1 for word, len(...) for char
                        chars_in_gap += len(words_to_chars(it, encode_back=True)) 
                        words_in_gap += 1

            accumu = self.i  ## left boundary
            waccumu = self.wi
            for i, c_gap, w_gap in vars:
            ##for sub in ded.ants:
                sub = ded.ants[i]
                if id(sub) not in nodememo:
                    sub.traverse(accumu + c_gap, waccumu + w_gap, fsent, rules, nodememo)
                    # accumulating # of edges (if first seen)
                    nedges += nodememo[id(sub)][1]

                ## don't accumulate subs now; will do in another visit
##                s += subs
                accumu = sub.j
                waccumu = sub.wj

            tmp_j = (ded.ants[vars[-1][0]].j if vars != [] else self.i) + chars_in_gap
            if self.j is not None and self.j != tmp_j:
                assert False, "@sentence %d, node #%s, %d %d != %d %s rule %d" % \
                       (opts.sentid, self.nodeid, self.i, self.j, tmp_j, self.x, ded.ruleid)
            self.j = tmp_j

            tmp_wj = (ded.ants[vars[-1][0]].wj if vars != [] else self.wi) + words_in_gap ##
            self.wj = tmp_wj
                
        self.id = len(nodememo) + 1
        nodememo[id(self)] = (self.id, nedges)



    def _dump(self, rules=None, nodememo=None, rulememo=None, deriv=None):

        if nodememo is None:
            nodememo = set()

        if rulememo is None:
            rulememo = set()

        if id(self) in nodememo:
            return ""

        nodememo.add(id(self))

        deds = self.deds
        
        # id, label, span, # of hyperedges
        s = "%d\t%s [%d-%d]\t%d\n" % (self.id, self.x, self.i, self.j, len(deds))  

        subs = ""
        # post-printout; hyperedges
        oracle = False
        
        for i, ded in enumerate(deds):
            outs = []
                
            for child in ded.lhsstr:
                if isinstance(child, Item):
                    out = child.id  ## cached id number
                    subs += child._dump(rules, nodememo, rulememo, deriv)
                else:
                    # english word, already quoted
                    out = child
                    
                outs.append(out)

            ruleid = ded.ruleid

            s += "\t"
            if id(self) in deriv.ded and deriv.ded[id(self)] == ded:
                ## this is an oracle edge, mark it
                s += "*"
                oracle = True
                
            s += "%s" % " ".join([str(x) for x in outs]) + \
                 " ||| %d" % ruleid
            if rules is None:
                pass ## no rule file supplied
            elif redundant_rules or ruleid not in rulememo:   ## asserting ruleid in rules
                rulememo.add(ruleid)
                s += " " + rules[ruleid]

            ## dcost is an svector.Vector (pyx)

            ## remove redundant features due to oracle computing
            for xx in ["cand", "src", "ref"]:
                del ded.dcost["oracle.%slen" % xx]
            for xx in range(4):
                del ded.dcost["oracle.match%d" % xx]
                
            s += " ||| %s\n" % (ded.dcost * trim_weights if slim_features else ded.dcost)

        if id(self) in deriv.ded:
            assert oracle, "Oracle derivation broken -- oracle hyperedge not found for node %" % \
                   self            

        return subs + s
            
        
    
class Deduction(object):
    '''In an and/or graph, this is an and node'''
    __slots__ = "rule", "ants", "dcost", "viterbi",   "ruleid", "lhsstr"  #lhuang
    def __init__(self, ants, rule, dcost=0.0, viterbi=None):
        self.ants = ants
        self.rule = rule
        self.dcost = dcost
        self.viterbi = viterbi

    def __str__(self):
        return str(self.rule)

    # Pickling
    def __reduce__(self):
        return (Deduction, (self.ants, self.rule, self.dcost))

### The functions below take a list of Items. They assume that the items are in topological
### order and that the last one is the root.                

def normalize_forest(chart):
    """Adjusts the forest so that all the OR nodes are proper probability distributions, such that
    the global probability distribution is normalized. The input dcosts are supplied as an argument;
    the output dcosts are left inside the Deductions. Requires inside probabilities, but does not update them!"""
    for item in chart:
        for ded in item.deds:
            p = ded.dcost
            for ant in ded.ants:
                p += ant.inside
            ded.dcost = p-item.inside

def compute_insideoutside(chart):
    compute_inside(chart)
    compute_outside(chart)

def compute_outside(chart):
    # chart is list of items, axiom first, goal last
    # requires inside probs

    for item in chart:
        item.outside = None
    chart[-1].outside = 0.0

    for item in reversed(chart):
        if item.outside is None:
            item.outside = cost.IMPOSSIBLE
            continue
        for ded in item.deds:
           if len(ded.ants) > 0:
                p = ded.dcost + item.outside
                for ant in ded.ants:
                    p += ant.inside
                for ant in ded.ants:
                    if ant.outside is None:
                        ant.outside = p-ant.inside
                    else:
                        ant.outside = cost.add(ant.outside,p-ant.inside)

def compute_viterbi(topological):
    for item in topological:
        item.viterbi_ded = None
        for ded in item.deds:
            c = ded.dcost
            for ant in ded.ants:
                c += ant.inside # assume this is viterbi inside
            if item.viterbi_ded is None or c < item.inside:
                item.viterbi_ded = ded
                item.inside = c

def compute_viterbi_outside(topological):
    """requires compute_viterbi"""
    for item in topological:
        item.outside = cost.IMPOSSIBLE

    topological[-1].outside = 0.0

    for item in reversed(topological):
        for ded in item.deds:
            c = ded.dcost
            for ant in ded.ants:
                c += ant.inside # assume this is viterbi inside
            for ant in ded.ants:
                antcost = item.outside + c - ant.inside # assume this is viterbi outside
                if antcost < ant.outside:
                    ant.outside = antcost

def compute_ded_mass(chart):
    """requires inside and outside"""
    for item in chart:
        for ded in item.deds:
            p = ded.dcost + item.outside
            for ant in ded.ants:
                p += ant.inside
            ded.dcost = p

def compute_ded_expectation(chart):
    total = chart[-1].inside
    compute_ded_mass(chart)
    for item in chart:
        for ded in item.deds:
            ded.dcost -= total

### Reading/writing forests in ISI format

def forest_to_text(f, mode=None, weights=None):
    result = []
    _item_to_text(f, result, {}, mode=mode, weights=weights)
    return "".join(result)

def _item_to_text(node, result, memo, mode=None, weights=None):
    if id(node) in memo:
        ## lhuang: already visited this node in the topological order
        result.append(memo[id(node)])
        return

    # lhuang: new node
    nodeid = len(memo)+1
    ##memo[id(node)] = "#%s" % nodeid
    memo[id(node)] = "%s" % nodeid # lhuang: nodeid does not have #

    ## keep only the top ten deductions to slim the forest
    deds = [(ded.dcost.dot(weights), ded) for ded in node.deds]
    deds.sort()

    for _, ded in deds[:10]:
        result.append('\n')
        _ded_to_text(ded, result, memo, mode=mode, weights=weights)

def _ded_to_text(node, result, memo, mode=None, weights=None):
    # Convert rule and features into single tokens
    #vstr = ",".join("%s:%s" % (quotefeature(f),node.dcost[f]) for f in node.dcost)
    # lhuang: in case no weights
    vstr = "cost:%s" % weights.dot(node.dcost) if weights is not None \
           else "_"
    rstr = id(node.rule)
    #rstr = id(node)
    s = "ruleid=%s<value=%s>" % (rstr,vstr)
    print "\truleid=%s" % rstr,
    
    if False and len(node.ants) == 0: # the format allows this but only if we don't tag with an id. but we tag everything with an id
        result.append(s)
    else:
        result.append('(')
        result.append(s)
        if mode == 'french':
            children = node.rule.f if node.rule else node.ants
        elif mode == 'english':
            # lhuang: default mode: english side
            children = node.rule.e if node.rule else node.ants
        else:
            children = node.ants

        for child in children:
            if isinstance(child, Item):
                result.append(' it ')
                _item_to_text(child, result, memo, mode=mode, weights=weights)
            elif sym.isvar(child):
                # lhuang: variable, do recursion
                result.append(' var ')
                _item_to_text(node.ants[sym.getindex(child)-1], result, memo, mode=mode, weights=weights)
            else:
                # lhuang: english word
                result.append(' word ')
                w = quoteattr(sym.tostring(child))
                result.append(w)
                print w,
        result.append(')')

    print # end of a hyperedge

class TreeFormatException(Exception):
    pass

dummylabel = sym.fromtag("-")
dummyi = dummyj = None

whitespace = re.compile(r"\s+")
openbracket = re.compile(r"""(?:#(\d+))?\((\S+)""")
noderefre = re.compile(r"#([^)\s]+)")
labelre = re.compile(r"^(-?\d*)(?:<(\S+)>)?$")

def forest_lexer(s):
    si = 0
    while si < len(s):
        m = whitespace.match(s, si)
        if m:
            si = m.end()
            continue

        m = openbracket.match(s, si)
        if m:
            nodeid = m.group(1)
            label = m.group(2)

            if label == "OR":
##                print >> logs, label, nodeid, 
                yield ('or', nodeid)
            else:
                m1 = labelre.match(label)
                if m1:
                    ruleid = m1.group(1)
                    vector = m1.group(2)
                    yield ('nonterm', nodeid, ruleid, vector)
                else:
                    raise TreeFormatException("couldn't understand label %s" % label)
            
            si = m.end()
            continue

        if s[si] == ')':
            si += 1
            yield ('pop',)
            continue

        m = noderefre.match(s, si)
        if m:
            noderef = m.group(1)
            yield ('ref', noderef)
            si = m.end()
            continue

        if s[si] == '"':
            sj = si + 1
            nodelabel = []
            while s[sj] != '"':
                if s[sj] == '\\':
                    sj += 1
                nodelabel.append(s[sj])
                sj += 1
            nodelabel = "".join(nodelabel)
            yield ('term', nodelabel)
            si = sj + 1
            continue

def forest_from_text(s, delete_words=[]):
    tokiter = forest_lexer(s)
    root = forest_from_text_helper(tokiter, {}, want_item=True, delete_words=delete_words).next()
    # check that all tokens were consumed
    try:
        tok = tokiter.next()
    except StopIteration:
        return root
    else:
        raise TreeFormatException("extra material after tree: %s" % (tok,))

def forest_from_text_helper(tokiter, memo, want_item=False, delete_words=[]):
    """Currently this assumes that the only frontier nodes in the tree are words."""
    while True:
        try:
            tok = tokiter.next()
            toktype = tok[0]
        except StopIteration:
            raise TreeFormatException("incomplete tree")

        if toktype == "or":
            _, nodeid = tok
            deds = list(forest_from_text_helper(tokiter, memo, \
                                                delete_words=delete_words))
            node = Item(dummylabel, dummyi, dummyj, deds=deds)
            if nodeid:
                memo[nodeid] = node
                node.nodeid = nodeid
            yield node

        elif toktype == "nonterm":
            _, nodeid, ruleid, dcoststr = tok
            if ruleid == "":
                ruleid = dummylabel
            else:
                # lhuang: N.B.: sym.fromtag would re-alloc it
                xrs_ruleid = int(ruleid)
                ruleid = sym.fromtag(ruleid)  #int(ruleid) #
                
            dcost = svector.Vector()
            if dcoststr:
                # lhuang: features are read from forest, not rules
                # so there is no "e^..." or "10^..."
                
                for fv in dcoststr.split(','):
                    f,v = fv.split(':',1)
                    v = float(v)
                    dcost[f] = v

            ants = []
            rhs = []
            vi = 1
            for child in forest_from_text_helper(tokiter, memo, want_item=True,\
                                                 delete_words=delete_words):
                if isinstance(child, Item):
                    ants.append(child)
                    rhs.append(sym.setindex(dummylabel, vi))
                    vi += 1
                else:
                    rhs.append(child)
            r = rule.Rule(ruleid, rule.Phrase(rhs), rule.Phrase(rhs))

            node = Deduction(ants=ants, rule=r, dcost=dcost)
            node.ruleid = xrs_ruleid
            
            if want_item: # need to insert OR node
                node = Item(dummylabel, dummyi, dummyj, deds=[node])
            if nodeid:
                memo[nodeid] = node
            yield node

        elif toktype == 'term':
            terminal = tok[1]
            if terminal not in delete_words:
                yield sym.fromstring(terminal)

        elif toktype == 'ref':
            yield memo[tok[1]]

        elif toktype == 'pop':
            return

        else:
            raise TreeFormatException("unknown token %s" % (tok,))

def read_rules(filename, rules=None):

    if rules is None:
       rules = {} 

    foreign_start_tag = '"' + foreign_sentence_tag + '" '

    getid = re.compile(r'(^|\s)id=(-?\d+)(\s|$)') # N.B.: boundaries!!
    # otherwise, sid=... might be confused with id=...
    
    ## rules like: VP-C-1(x0:VB-1 x1:NP-0) -> x0 x1 ### count_category_5=10^-1 ... id=181489313 ...
    for i, line in enumerate(open(filename).xreadlines()):

        if line == "\n": ## not ""!
            break   # end of a segment (finished rules for one sentence)
        
        rule, feats = line.split(" ###", 1) ## must leave a space

        try:
            ruleid = int(getid.findall(feats)[0][1]) ## (^|\s) ... (\s|$)
        except:
            assert False, "@sentence %d, BAD RULE LINE@%d:\t%s" % (opts.sentid, i, line)
            
        rules[ruleid] = rule.replace(foreign_start_tag, "") if not output_foreign_start else rule

    print >> sys.stderr, "%d rules read" % len(rules)

    rules[0] = "TOP() -> " # N.B.: fields stripped off already
    
    return rules

def read_weights(f):

    # one-line format: 's_insertion:-0.141093,_lrb__nodes:0.315827,...

    pair = [x.split(":") for x in f.readline().split(",")]
    features = {}
    weights = {}
    for feat, value in pairs:
        fid = len(features) + 1 
        features[feat] = fid   # one-based index
        weights[feat] = float(value)  # indexed on id, not name!

    return weights        

# from hope.py
def output(node, prompt, gbleu, gscore):
    deriv = node.viterbi_deriv()
    hyp = " ".join([sym.tostring(e) for e in deriv.english()])
    bleu = fbleu.rescore(hyp)
    score = weights.dot(deriv.vector())

    # in place!!
    gbleu += fbleu.copy()
    gscore += [score]

    print >> logs,  "%s  \tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\n%s" % \
          (prompt, score, bleu, fbleu.ratio(), hyp)            


if __name__ == "__main__":

    import optparse
    optparser = optparse.OptionParser(usage="usage: cat <isi-forests> | %prog [options] > <my-forests>")
    ## default value for k is different for the two modes: fixed-k-best or \inf-best
    optparser.add_option("-k", "", dest="k", type=int, help="k-best", metavar="K", default=None)
    optparser.add_option("", "--id", dest="sentid", type=int, help="sentence id", metavar="ID", default=1)
    ## weights
    optparser.add_option("-w", "--weights", dest="weights", type=str, help="weights file or str", metavar="WEIGHTS", default="lm1=2 gt_prob=1")
    optparser.add_option("-f", "--foreign", dest="foreign", type=str, help="source file", metavar="FILE")
    optparser.add_option("-b", "--byline", dest="byline", type=str, help="byline file", metavar="FILE")
    optparser.add_option("-r", "--rules", dest="rules", type=str, help="rules file", metavar="FILE")
    optparser.add_option("-e", "--extrarules", dest="extrarules", type=str, help="extra (global) rules file", metavar="FILE")
    optparser.add_option("", "--oracle", dest="compute_oracle", action="store_true", help="compute oracles", default=False)
    optparser.add_option("", "--hope", dest="hope", type=float, help="hope weight", default=0)

    (opts, args) = optparser.parse_args()

    from forest import get_weights
    weights = get_weights(opts.weights)

    # should have a special "Identity" vector (11...1)
    trim_weights = 1 # svector.Vector("gt_prob=1")
    slim_features = False
    redundant_rules = False
    output_foreign_start = False
    foreign_sentence_tag = "<foreign-sentence>"
    max_edges_per_node = 1000000
    start_sent_id = 1

    # command-line: cat <isi-forest> | ./convert_forest_to_my.py <rules> <f_sent> <bylines> <refs>+
    import monitor
    sys.stderr.write("t=%s start\n" % monitor.cpu())
##    if len(sys.argv) < 2:
##        print >> sys.stderr, "WARNING: no rule files supplied -- output forest" \
##              + "will contain ruleids only"
##        rules = None
##    else:

    
    forestfile = sys.stdin
    srcfile = open(opts.foreign)
    bylinefile = open(opts.byline)
    reffiles = [open(f) for f in args] ## the remaining of the input are assumed to be refs


##    print >> logs, "rules file %s" % rulefile
##    print >> logs, "source file %s" % srcfile
##    print >> logs, "byline file %s" % bylinefile
##    print >> logs, "re files %s" % " ".join(map(str, reffiles))

    # lhuang: n-gram order = 4
    theoracle = oracle.Oracle(4, variant="ibm")

    hopebleus = collections.defaultdict(lambda : Bleu())
    hopescores = collections.defaultdict(lambda : [])
    
    onebestbleus = Bleu()
    onebestscores = []
    
    for i, (srcline, byline, forestline) in \
            enumerate(itertools.izip(srcfile, bylinefile, forestfile)):

        reflines = [f.readline() for f in reffiles]

        rules = read_rules(opts.rules)
        if opts.extrarules:
            rules = read_rules(opts.extrarules, rules)

        if forestline.strip() == "": ## empty forest (pure byline)
            forestline = "(0<gt_prob:0> )"
        f = forest_from_text(forestline)
        

        # for bleu
        fbleu = Bleu(refs=reflines)

        srcline = srcline.split()
        if srcline[0] == foreign_sentence_tag:
            del srcline[0]
        srcline = " ".join(srcline)
        
        print f.dump(rules, \
                     i + opts.sentid, \
                     srcline, byline.strip(), reflines)
        sys.stderr.write("t=%s read line %s\n" % (monitor.cpu(), i+1))
        sys.stderr.flush()

        if i % 10 == 9:
            print >> logs,  "overall 1-best bleu = %.4lf (%.2lf) score = %.4lf" \
                  % (onebestbleus.score_ratio() + (sum(onebestscores)/(i+1),))
            for key in hopebleus:
                print >> logs,  "overall hope%s   bleu = %.4lf (%.2lf) score = %.4lf" \
                      % (key, hopebleus[key].score_ratio() + (sum(hopescores[key])/(i+1),))
