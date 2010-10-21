#!/usr/bin/env python

''' Feature -> FeatureClasses
but individual features are just strings
'''


import sys
import os
import copy
##import pp

##job_server = pp.Server(2, ppservers=())
## syntactic head = 1, semantic head = 0

use_pp = False

#modelsdir = os.environ["MODELS"] 

logs = sys.stderr

probweight = -0.283551   # weight of the first feature, the logprob from charniak parser

from utility import quantize, bound_by, desymbol, make_punc   # johnson's quantize function, in this dir
from tree import *
from svector import Vector

cross_lines = False  # output format

class Feature(object):

    _classes = {}

    NODELOCAL = 0
    EDGELOCAL = 1
    NONLOCAL = 2
    GLOBAL = 3

    LEFT = 0
    RIGHT = 1

    dir_markers = ["*LEFT*", "*RIGHT*"]
    end_marker = "_"
    adj_markers = ["*NONADJ*", "*ADJ*"]


    @staticmethod
    def get_label(node):
        return node.label if node is not None else Feature.end_marker

    @staticmethod
    def get_label_head(node, head):
        if node is not None:
            return node.label + ("!" if node is head else "")
        else:
            return Feature.end_marker

    def onecount(self, f, sub_cat=0):
        ## new connector: @; also no ", 1"
        return ("%s~%s" % (self._str[sub_cat], f)) #, 1)

    def __init__(self):
        self.hashval = None            
        
    def __hash__(self):
        if self.hashval is None:
            self.hashval = hash(str(self))
        return self.hashval
    
    def __str__(self):
        return self._str[0] ## must be set up on __init__

    __repr__ = __str__
    
    def extract(self, tree, sentence):
        '''count the counts of features of this class on this (sub-) tree'''
        pass

    def is_local(self):
        return self._locality <= Feature.EDGELOCAL

    def is_nonlocal(self):
        return not self.is_local()
    
    def is_global(self):
        return self._locality == Feature.GLOBAL

    def is_nodelocal(self):
        return self._locality == Feature.NODELOCAL

    def is_edgelocal(self):
        return self._locality == Feature.EDGELOCAL        

class Rule(Feature):

    '''Rule:0:<nanccats>:<root>:<conj>:<head>:<functional>:0:1 '''

    ### example: TOP -> FRAG -> A -> (B, C)
    ### format: (<curr_NTs> _ [<parent_NT> [<grand_parent_NT>]])
    ### rule-1: (B C _ A)        (A _ FRAG)    (FRAG _)
    ### rule-2: (B C _ A FRAG)   (A _ FRAG)    (FRAG _)

    ## N.B.: level=1 or 2
    
    def __init__(self, level=1, root=False, conj=False, head=0, functional=0):
        Feature.__init__(self)
        self._locality = Feature.EDGELOCAL if level==1 else Feature.NONLOCAL
        
        self.level = level
        self.root = root
        self.conj = conj
        self.head = head
        self.functional = functional
        self._str = ["Rule:0:%d:%d:%d:%d:%d:0:1" % \
                    (self.level-1, self.root, self.conj, self.head, self.functional)]

    def make(self, subs, label, parentlabel=None):
        ''' return something like (VBP ADVP PP , S _ VP) for rule-1 
        and (RB RB VP _ VP VP) for rule-2 '''

        elements = [sub.label for sub in subs] + ["_"]        
        if label not in ["TOP", "S1"]:
            elements.append(label)
            if self.level !=1 and parentlabel not in [None, "TOP", "S1"]:
                elements.append(parentlabel)

##        print self.level, label, parentlabel, elements
        return self.onecount("(%s)" % " ".join(elements))
    
    def extract(self, tree, sentence):

        if self.level == 1:            
            return [self.make(tree.subs, tree.label)] if not tree.is_terminal() else []
        else:
            ## special design converting non-local to edge-local
            a = [self.make(sub.subs, sub.label, tree.label) \
                 for sub in tree.subs if not sub.is_terminal()] \
                 if not tree.is_terminal() else []

            ## special care for TOP -> S: (S _)
            if tree.is_root():
                a.append(self.make(tree.subs, tree.label))

            return a
            


class Word(Feature):

    def __init__(self, level):
        ''' level = 1 or 2 (Word:1 vs. Word:2)'''
        Feature.__init__(self)
        self._locality = Feature.NODELOCAL if level==1 else Feature.EDGELOCAL

        self.level = level
        self._str = ["Word:%d" % self.level]

    def make(self, word, label, parent=None):
        elements = [word, label]
        if self.level == 2:
            elements.append(parent)
        return self.onecount("(%s)" % " ".join(elements))

    def extract(self, tree, sentence, dep=0):

        if self.level == 1:            
            return [self.make(tree.word, tree.label)] if tree.is_terminal() else []
        else:
            ## special design converting non-local to edge-local
            return [self.make(sub.word, sub.label, tree.label) \
                    for sub in tree.subs if sub.is_terminal()] \
                    if not tree.is_terminal() else []


##class BadHeavy(Feature):
##    ''' WRONG IMPLEMENTATION OF HEAVY. replaced by the correct one below.'''
##    ## 425858  Heavy ((5 4) (PP '' _))
##    ## (binned_len binned_distance_to_end) (label final_punc following_punc)
##    ## note: puncs mean words, not POS tags

##    def __init__(self):
##        self._str = "Heavy"

##    def make(self, binned_len, distance, label, final_punc, final_word):
##        return self.onecount("((%d %d) (%s %s %s))" % (binned_len, distance, label, final_punc, follow_punc))
    
##    def extract(self, tree, sentence):
##        if tree.is_terminal():
##            return []
        
##        binned_len = tree.binned_span_width()
##        distance = quantize(len(sentence) - tree.span[1])   ## will be moved into tree

##        ## will be integrated into tree by passing a sentence, but not storing the sentence
##        ## make_punc is in tree.py
        
##        final_punc = make_punc(sentence[tree.span[1] - 1])   
##        follow_punc = make_punc(sentence[tree.span[1]]) if tree.span[1] < len(sentence) else "_"

##        return [self.make(binned_len, distance, tree.label, final_punc, follow_punc)]

    
class Heavy(Feature):
    ''' CORRECT IMPLEMENTATION OF HEAVY. it is a unbounded feature now.
    Algorithm:
    For each node t,
         for all t\'s non-right-most child sub,
              trace sub\'s right-most path
    also store tag_seq at each node
    globally-right-most are popped ASAP.
    '''
    ## 425858  Heavy ((5 4) (PP '' _))
    ## (binned_len binned_distance_to_end) (label final_punc following_punc)
    ## note: puncs mean words, not POS tags

    def __init__(self):
        Feature.__init__(self)
        self._locality = Feature.NONLOCAL
        
        self._str = ["Heavy"]

    def make(self, tree, final_tag_word, follow_tag_word):
        # in utility.py
        final_punc = make_punc(final_tag_word)   
        follow_punc = make_punc(follow_tag_word)

        return self.onecount("((%d %d) (%s %s %s))" % (tree.binned_len(), \
                                                  tree.distance, \
                                                  tree.label, \
                                                  final_punc, follow_punc))
    
    def extract(self, tree, sentence):
        if tree.is_terminal():
            tree.waitings = []
            return []
        
        tree.distance = quantize(len(sentence) - tree.span[1])   ## will probably be moved into tree

        a = []
        # for all non-right-most children
        for i, sub in enumerate(tree.subs[:-1]):
            # get its right sibling's first (postag, word) pair
            right_pos_word = tree.subs[i+1].get_tag_word(0) 

            for waiting in sub.waitings:
                a.append(self.make(waiting, waiting.get_tag_word(-1), right_pos_word))

        if tree.distance == 0: # i am a right-most constituent
            a.append(self.make(tree, tree.get_tag_word(-1), ("_", "_")))
        else:
            ## there is still stuff to my right, so wait for my parents
            tree.waitings = tree.subs[-1].waitings + [tree]
            
        return a    

def remove_split(label):
    '''remove berkeley state-split from nonterminal labels.
       VP-0 => VP, @SBAR-1-BAR => @SBAR-BAR.'''
    if label[-4:] == "-BAR":
        bar, label = "-BAR", label[:-4]
    else:
        bar = ""

    p = label.rfind("-")
    if p >= 0 and label[p+1:].isdigit():
        label = label[:p]

    return label+bar        
    
class WordEdges(Feature):
    ''' a local feature! '''
    
    def __init__(self, use_len=False, leftprec=0, leftsucc=0, rightprec=0, rightsucc=0):
        Feature.__init__(self)
        self._locality = Feature.NODELOCAL
        
##        self._str = ["WordEdges:%d:%d:%d:%d:%d" % (use_len, leftprec, leftsucc, rightprec, rightsucc)]
        # TODO: binarized format (or trinarized)
        # actually, characters, so "CE"
        self._str = ["CE%d~%d%d%d%d" % (use_len, leftprec, leftsucc, rightprec, rightsucc)] # N.B. ":"->"~"

        self.use_len = use_len
        self.leftprec = leftprec
        self.leftsucc = leftsucc
        self.rightprec = rightprec
        self.rightsucc = rightsucc

    def make(self, tree, seq):
        s = "%d^" % tree.binned_len() if self.use_len else ""
        s += remove_split(tree.label)
        if len(seq) > 0:
            s += "^" + "^".join(seq)
        return self.onecount("%s" % s)
        
    def extract(self, tree, sentence):
        if tree.is_terminal():
            return []

        seq = []
        ## append 2 _'s at the end of a *new copy* of the sentence.
        ## do not override the old sentence, which is used elsewhere also.

        sent = sentence + ["_", "_"] ## caution, TODO: should be generic
        
        for i in xrange(1, self.leftprec + 1):
            seq.append(sent[tree.span[0] - i])

        for i in xrange(1, self.leftsucc + 1):
            seq.append(sent[tree.span[0] - 1 + i])

        for i in xrange(1, self.rightprec + 1):
            seq.append(sent[tree.span[1] - i])

        for i in xrange(1, self.rightsucc + 1):
            seq.append(sent[tree.span[1] - 1 + i])

        return [self.make(tree, seq)]    
                


##class WProj(Feature):
##    '''// Identifier is WProj:<HeadType>:<IncludeNonMaximal>:<NAncs>
##       // lhuang: only one type: WProj:0:0:1 -- semantic-head, non-off-spans, 1 outside NT
##       e.g.  WProj:0:0:1 (unclear ADJP VP)   WProj:0:0:1 (said SINV S1) 
##       ADJP is the maximal projection of the word "unclear," while VP is the immediate NT above ADJP.
##       S1 is the TOP.

##       NOT-IMPLEMENTED: including off-spans, more outside NTs
##       '''
    
##    def __init__(self, htype=heads.SEM):
##        Feature.__init__(self)
##        self._locality = Feature.NONLOCAL

##        self.htype = htype
##        self._str = ["WProj:%d:0:1" % htype]

##    def make(self, word, (maximal, outside)):
##        return self.onecount("(%s %s %s)" % (word, maximal, outside))

##    def extract(self, tree, sentence):
##        '''old implementation: local, with top-down annotations;
##           new implementation on-the-fly bottom-up:
##              for non-heads:
##                 make
##           '''
##        a = []

##        if not tree.is_terminal():
##            head = tree.get_headchild(self.htype)
##            for sub in tree.subs:
##                if sub is not head or tree.is_root():
##                    h = sub.get_lexhead(self.htype)
##                    assert h is not None, "tree=%s sub=%s" % (tree, sub.labelspan())
##                    if not h.is_punctuation():
##                        a.append(self.make(h.word, (sub.label, tree.label)))
##        return a
    
####        if tree.is_terminal() and not tree.is_punctuation():
####            return [self.make(tree.word, tree.get_maximal_outside(self.htype))]
####        else:
####            return []
        

##class Heads(Feature):
##    '''Heads: head-dependent pairs/triples
##       NOT-IMPLEMENTED: headlex != deplex
##       e.g.: (head is always to the right)
##       Heads:2:1:1:0 (RP off VBN cut)   # RP (off) -> VB (cut)    
##       Heads:3:0:0:0 (JJ NNP VBD)       # JJ (black) -> NNP (Monday) -> VBD (was)
##       '''

##    def __init__(self, levels=2, headlex=True, deplex=True, htype=heads.SYN):
##        Feature.__init__(self)
##        self._locality = Feature.NONLOCAL

##        self.levels = levels
##        self.headlex = headlex
##        self.deplex = deplex
##        assert (headlex == deplex), "headlex != deplex: NOT IMPLEMENTED."
##        self.htype = htype
##        ## N.B.: in WProj, Johnson is sem-syn, in Heads, he is syn-sem! inconsistent!
##        self._str = ["Heads:%d:%d:%d:%d" % (levels, headlex, deplex, 1-htype)]

##    def make(self, nodes):
##        assert len(nodes) == self.levels, "levels = %d, len(nodes) = %d" % (self.levels, len(nodes))
##        if self.headlex:
##            s = " ".join(["%s %s" % (node.label, node.word) for node in nodes])
##        else:
##            s = " ".join(["%s" % node.label for node in nodes])

##        return self.onecount("(%s)" % s)

##    def extract(self, tree, sentence):
##        htype = self.htype
##        if tree.is_terminal():
                
##            tree.allheads[htype] = [heads.LexHead(tree)]
##            tree.ccheads[htype] = [heads.LexHead(tree)]
##            tree.twolevels[htype] = []
##            return []

##        if tree.is_root():
##            return []

##        if tree.is_coordination():
##            ## coord-node
##               tree.allheads[htype] = []
##               tree.ccheads[htype] = []
##               tree.twolevels[htype] = []

##            for sub in tree.subs:
##                tree.allheads[htype].extend(sub.allheads[htype])    
##                tree.twolevels[htype].extend(sub.twolevels[htype])    
##                if sub.label == tree.label: ## LH: this is wrong of MJ!
##                    tree.ccheads[htype].extend(sub.ccheads[htype])

##            return []
        
##        else:
##            ## normal node

##            a = []

##            head = tree.get_headchild(htype)
##            assert head is not None, "%s's %s-head is None!" % (tree, htype)

##            ##print "at %s the %s-head is %s" % (tree.spanlabel(), htype, head.label)

##            tree.allheads[htype]  = copy.copy(head.allheads[htype])
##            tree.ccheads[htype]   = copy.copy(head.ccheads[htype])
##            ## N.B.: when you attach, include head's own!
##            tree.twolevels[htype] = copy.copy(head.twolevels[htype])
            

##            ##print tree.labelspan(), "initial two levels for htype", htype, tree.twolevels[htype]
##            ##print tree.labelspan(), "head = ", head, id(head)
##            for sub in tree.subs:
##                if sub is not head:
##                    for dep in sub.allheads[htype]:
##                        for gov in head.ccheads[htype]:
##                            if self.levels == 2:
##                                a.append(self.make((dep, gov)))
##                            else:
##                                tree.twolevels[htype].append((dep, gov))

##            ##print tree.labelspan(), "final two levels for htype", htype, tree.twolevels[htype]

##            if self.levels == 2:
##                return a
##            else:
##                ## 3-levels
##                a = []
##                for sub in tree.subs:
##                    if sub is not head:
##                        ##print sub.labelspan(), "two levels for htype", htype, sub.twolevels[htype]
##                        for deps in sub.twolevels[htype]:
##                            for gov in head.ccheads[htype]:
##                                ##print tree.labelspan(), deps, gov
##                                a.append(self.make(deps + (gov,)))
##                return a
        

##class HeadTree(Feature):

##    def __init__(self, lex=False, htype=heads.SYN):
##        self._str = ["HeadTree:%d:%d:%d:%d" % (1, lex, 0, 1-htype)]
##        self._locality = Feature.NONLOCAL

##        self.lex_htype = (lex, htype)
##        self.lex = lex
##        self.htype = htype

##    def make(self, s):
##        if s.find(" ") >=0 :
##            s = "\"%s\"" % s
##        return self.onecount(s)

##    def extract(self, tree, sentence):

##        if tree.is_terminal():
##            ## use % instead of \% here, WHY???!
##            word = tree.word.replace("\%", "%")
##            tree.headspath[self.lex_htype] = "(%s %s)" % (tree.label, word) \
##                                             if self.lex else tree.label
##            return []

##        else:
##            head = tree.get_headchild(self.htype)
###            print "visit ", tree.spanlabel(), ", head=", head.spanlabel(), ", path=", head.headspath[self.htype]
##            a = []
##            thislevel = []
##            for i, sub in enumerate(tree.subs):
##                if sub is not head:
##                    a.append(self.make(sub.headspath[self.lex_htype]))
##                    if i > 0 and head is tree.subs[i-1] or \
##                       i < len(tree.subs) - 1 and head is tree.subs[i+1]:
##                        ## only immediate left/right siblings of the head
##                        thislevel.append(sub.label)
##                else:
##                    thislevel.append(sub.headspath[self.lex_htype])

##            tree.headspath[self.lex_htype] = "(%s %s)" % (tree.label, " ".join(thislevel))

##            if tree.is_root(): ## in case of root/S1
##                return [self.make(tree.headspath[self.lex_htype])]
            
##            return a
        

class RightBranch(Feature):

    def __init__(self):
        Feature.__init__(self)
        self._str = ["RightBranch"]
        self._locality = Feature.GLOBAL

    def make(self, a, counta):
        ''' this is different from most other features that are unit-valued.'''
        return ["%s %d" % (self._str[0], a)] * counta if counta > 0 else []

    @staticmethod
    def count(trees, rightmost, counts):
        ''' this is almost verbatim from Johnson\'s spfeatures.h. see RightBranch class in spfeatures.h
            in order to replicate it, i simply "transcribed" it,
            which actually pays off -- preterminals are INCLUDED (his slides were wrong)
        '''
        if len(trees) > 1:
            rightmost = RightBranch.count(trees[1:], rightmost, counts)
        node = trees[0]
        if node.is_punctuation():
            return rightmost
        counts[rightmost] += 1
        if not node.is_terminal():
            RightBranch.count(node.subs, rightmost, counts)
        return 0

    def extract(self, tree, sentence):
        assert tree.is_root()
        counts = {1:0, 0:0}
        RightBranch.count([tree], 1, counts)
        ## N.B. if non-right-branch = 0 (i.e., monadic), don't include it
        return self.make(1, counts[1]) + self.make(0, counts[0])
    

class CoLenPar(Feature):
    
    def __init__(self):
        Feature.__init__(self)        
        self._str = ["CoLenPar"]
        self._locality = Feature.EDGELOCAL

    def make(self, dist, is_final):
        return self.onecount("(%d %d)" % (dist, is_final))

    def extract(self, tree, sentence):
        a = []
        if tree.is_coordination():
            last_sub = None
            for i, sub in enumerate(tree.subs):
                if not sub.is_punctuation() and not sub.is_conjunction():  ## i am a conjunct
                    size = sub.span_width()
                    if last_sub is not None:
                        d = size - last_size
                        a.append(self.make(bound_by(d, 5), i == len(tree.subs)-1))  ## |d| <=5, is_final?
                        
                    last_sub = sub
                    last_size = size
                    
        return a

class NGramTree(Feature):

    def __init__(self, ngram=2, lex=0):
        Feature.__init__(self)

        # processing 0 and 3 at the same time
        self._str = ["NGramTree:%d:%d:1:0" % (ngram, l) for l in [0, 3]]
        self._locality = Feature.NONLOCAL
        self.ngram = ngram
        self.lex = lex

    def out(self, path):
        '''from core outside (bottom-up)'''

        # s: unlex, t: lex
        s = path[0].label
        ## caution again, in HeadTree and NGramTree, % are %
        t = "(%s %s)" % (path[0].label, path[0].word.replace("\%", "%"))
        for node in path[1:]:
            s = "(%s %s)" % (node.label, s)
            t = "(%s %s)" % (node.label, t)
        return s, t
        
    def make(self, head, leftpath, rightpath):

        # unlex and lex versions
        lla, llb = self.out(leftpath)
        rra, rrb = self.out(rightpath)

        # s: unlex, t: lex
        s = "\"(%s %s %s)\"" % (head.label, lla, rra)
        t = "\"(%s %s %s)\"" % (head.label, llb, rrb)
##        print head.labelspan(), "\t", "%s-%s\t" % (leftpath[0].word, rightpath[0].word), s
        return [self.onecount(s, 0), self.onecount(t, 1)] 

    def extract(self, tree, sentence):

        a = [] 
        if tree.is_terminal():
            tree.leftmost = [tree]
            tree.rightmost = [tree]

        else:
            for i, left in enumerate(tree.subs[:-1]):

                right = tree.subs[i+1]
                ## do not include last punc (bug of johnson)
                if right.span[0] != len(sentence)-1:
                    a.extend(self.make(tree, left.rightmost, right.leftmost))

            tree.leftmost = tree.subs[0].leftmost + [tree]  ## bottom-up! 
            tree.rightmost = tree.subs[-1].rightmost + [tree]

        return a

class TagEdges(WordEdges):
    ''' a non-local feature; looks very similar to WordEdges; implementation similar to NGramTree.
    '''
    
    def __init__(self, use_len=False, leftprec=0, leftsucc=0, rightprec=0, rightsucc=0):
        WordEdges.__init__(self, use_len, leftprec, leftsucc, rightprec, rightsucc)
        self._locality = Feature.NONLOCAL        
        self._str = ["Edges:%d:%d:%d:%d:%d" % (use_len, leftprec, leftsucc, rightprec, rightsucc)]

    def extract(self, tree, sentence):
        if tree.is_terminal():
            return []

        if self.leftprec == 0 and self.rightsucc == 0:  ## no outward (easy case: all inside)
            if tree.span_width() >= max(self.leftsucc, self.rightprec):
                seq = tree.tag_seq[:self.leftsucc] + \
                      list(reversed(tree.tag_seq[len(tree.tag_seq) - self.rightprec:]))
                return [self.make(tree, seq)]
            else:
                return []

##        else:
##            prev_tag = [tree.subs[i-1].tag_seq[-1]] if i > 0 else []

##            for i, sub in enumerate(tree.subs):
##                if self.rightsucc > 0 and i < len(tree.subs) - 1:
##                next_tag = tree.subs[i+1].tag_seq[0]                
##                for node in sub.rightmosts:   # all the waiting nodes on the right most branch
##                    seq = prev_tag + \
##                          node.tag_seq[:self.leftsucc] + node.tag_seq[-self.rightprec:] + \
##                          next_tag
                    
##                    a.append(self.make(node, seq))
                
                    
##            tree.rightmosts = tree.subs[-1].rightmosts + [tree]
##            tree.leftmosts = tree.subs[0].leftmosts + [tree]
            
            
            
                

#####################################################

# collins style features
#
# Bigram, Trigram, HeadMod, and DistMod from (Collins and Koo, 2003)
# 
#####################################################

##class Trigram(Feature):

##    def __init__(self, lex=0, htype=heads.SYN):
##        '''rule           VP -> PP VBD NP SBAR
##           will produce  (VP _ PP VBD!)    (VP PP VBD! NP)   (VP VBD! NP SBAR)     (VP NP SBAR _)
##        '''
        
##        self._str = ["Trigram:%d:%d" % (lex, htype)]
##        self._locality = Feature.EDGELOCAL if (lex == 0) else Feature.NONLOCAL
##        self.lex = lex
##        self.htype = htype

##    def make(self, tree, sub, prev, next, head):
##        labels = map(lambda x: Feature.get_label_head(x, head), [prev, sub, next])
##        return self.onecount("(%s %s)" % (tree.label, " ".join(labels)))

##    def extract(self, tree, sentence):
##        if tree.is_terminal():  ## even unary rules do
##            return []

##        head = tree.get_headchild(self.htype)
##        a = []
##        ## left
##        subs = tree.subs[:] + [None]  ## so that -1 and n will be defined, automatically

##        direction = Feature.LEFT
##        for i, sub in enumerate(tree.subs):
##            prev, next = subs[i-1], subs[i+1]
##            a.append(self.make(tree, sub, prev, next, head))

##        return a


##class Bigram(Feature):
##    ''' This feature will be inherited by HeadMod and DistMod,
##        all of which are two-body relations involving heads.
##    '''

##    def __init__(self, grandparent=0, lex=0, htype=heads.SYN, name="Bigram"):
##        '''grandparent = 0: nothing, 1: NT, 2: rule.
##           rule           VP -> PP VBD NP SBAR
##           will produce  (VP *LEFT* PP _)    (VP *RIGHT* NP SBAR)   (VP *RIGHT* SBAR _)    
##        '''
        
##        self._str = ["%s:%d:%d:%d" % (name, grandparent, lex, htype)]
##        self._locality = Feature.EDGELOCAL if (lex == 0 and grandparent == 0) else Feature.NONLOCAL
##        self.grandparent = grandparent
##        self.lex = lex
##        self.htype = htype

##    def make(self, direction, tree, sub, other, adj=0):
##        '''to be overridden by subclasses'''
        
##        mark = Feature.dir_markers[direction]
##        other_label = Feature.get_label(other)
##        return self.onecount("(%s %s %s %s)" % (tree.label, mark, sub.label, other_label))        

##    def dostuff(self, direction, a, tree, subs, head, i):
##        '''to be overriden by subclasses'''
        
##        sub = subs[i]
##        other = subs[i-1] if direction == Feature.LEFT else subs[i+1]
##        a.append(self.make(direction, tree, sub, other))

##    def extract(self, tree, sentence):
##        if tree.is_terminal() or len(tree.subs) == 1:
##            return []

##        head = tree.get_headchild(self.htype)

##        a = []
##        ## left
##        subs = tree.subs[:] + [None]  ## so that -1 and n will be defined, automatically

##        direction = Feature.LEFT
##        for i, sub in enumerate(tree.subs):
##            if sub is head:
##                ## turns the around way around :P
##                direction = Feature.RIGHT
##            else:
##                self.dostuff(direction, a, tree, subs, head, i)

##        return a


##class HeadMod(Bigram):

##    def __init__(self, grandparent=0, lex=0, htype=heads.SYN):
##        '''grandparent = 0: nothing, 1: NT.
##           rule           VP -> PP VBD NP SBAR
##           will produce  (VP *LEFT* VBD PP *ADJ*) (VP *RIGHT* VBD NP *ADJ*) (VP *RIGHT* VBD SBAR *NONADJ*)    
##        '''

##        Bigram.__init__(self, grandparent, lex, htype, name="HeadMod")

##    def make(self, direction, tree, sub, other, adj):
##        mark = Feature.dir_markers[direction]
##        other_label = Feature.get_label(other)
##        adj_mark = Feature.adj_markers[adj]
##        return self.onecount("(%s %s %s %s %s)" % (tree.label, mark, sub.label, other_label, adj_mark))        

##    def dostuff(self, direction, a, tree, subs, head, i):
##        '''to be overriden by subclasses'''
        
##        sub = subs[i]
##        adj = head is subs[i-1] or head is subs[i+1]
##        a.append(self.make(direction, tree, head, sub, adj))


##class DistMod(Bigram):

##    def __init__(self, grandparent=0, lex=0, htype=heads.SYN):
##        '''grandparent = 0: nothing, 1: NT.
##           N.B. simpler than Collins: distance will be quantized absolute dist. (0, 1, 2, 3-4, >=5)
##           rule           VP -> PP[1-4] VBD[4-5] NP[5-6] SBAR[6-15] .[15-16]
##           will produce  (VP VBD PP 0) (VP VBD NP 0) (VP VBD SBAR 1) (VP VBD . 5)
##        '''

##        Bigram.__init__(self, grandparent, lex, htype, name="DistMod")

##    def make(self, direction, tree, sub, other, dist):
##        other_label = Feature.get_label(other)
##        return self.onecount("(%s %s %s %d)" % (tree.label, sub.label, other_label, dist))        

##    def dostuff(self, direction, a, tree, subs, head, i):
##        '''quantized absolute distance (number of words in b/w). see utility.py'''
        
##        sub = subs[i]
##        if direction == Feature.LEFT:
##            dist = head.span[0] - sub.span[1]
##        else:
##            dist = sub.span[0] - head.span[1]
##        dist = quantize(dist)
##        a.append(self.make(direction, tree, head, sub, dist))

    
#####################################################

## main

#####################################################

def extract(tree, sentence, fclasses, do_sub=True, logprob=None):
    ''' extract all features, return a Vector.
        visit subtrees first, and then extract all features on this level.
        mapping from full-names to ids are in fvector.py

        the non-sub version (just this level is used by BUDecoder (forest decoder).
    '''    

    fvector = Vector()

    tree.annotate(None, do_sub=do_sub)
    
    if do_sub:
        if not tree.is_terminal():
            for sub in tree.subs:
                fvector += extract(sub, sentence, fclasses)

    jobs = []
    for fclass in fclasses:
        if not fclass.is_global() or tree.is_root():
            if use_pp:
                jobs.append(job_server.submit(fclass.extract, (tree, sentence), (quantize,)))
            else:
                fvector += Vector.convert_fullname(fclass.extract(tree, sentence))

    if use_pp:
        for job in jobs:
            fvector += Vector.convert_fullname(job())

    if logprob is not None:
        fvector[0] = logprob
    return fvector


def read_features(featurefilename, feature_mapping, reverse_mapping):

    ## 379926\tWord:1 (archrival NN)
    print >> logs, "reading features from %s" % featurefilename
    featurefile = open(featurefilename)
    
    idset = set([])
    i = 0
    for line in featurefile:
        Id, fstr = line.strip().split("\t")
        Id = int(Id)
        assert (Id not in idset), "duplicate feature id: %d => %s" % (Id, fstr)
        feature_mapping[fstr] = Id
        reverse_mapping[Id] = fstr
        idset.add(Id)
        i += 1

    print >> logs, "read %d features" % i
    
    return idset


def read_weights(weightsfilename, idset):

    j = i = len(idset)

    print >> logs, "reading weights from %s" % weightsfilename
    weightsfile = open(weightsfilename)
    ## 425858=0.0230297
    for line in weightsfile:
        Id, w = line.split("=")
        Id = int (Id)
        assert (Id in idset)
        weights [Id] = float(w)
        i -= 1

    assert (i == 0)

    print >> logs, "read %d weights" % j


def wordedges(max_words=2):
    wordedges = []
    for binlen in [True]: # [False, True]
        for leftprec in xrange(max_words + 1):
            for leftsucc in xrange(max_words + 1):
                for rightprec in xrange(max_words + 1):
                    for rightsucc in xrange(max_words + 1):
                        if leftprec + leftsucc + rightprec + rightsucc <= max_words:
                            f = WordEdges(binlen, leftprec, leftsucc, rightprec, rightsucc)
                            wordedges.append(f)

    return wordedges

def tagedges(max_inside=2, max_outside=0, zero_ok=False):
    ''' zero_ok=True would include redundant features from WordEdges, i.e.
        nonterminal/binlength only, with no tag/word info.    '''
    tagedges = []
    for binlen in [False, True]:
        for leftprec in xrange(max_outside + 1):
            for leftsucc in xrange(max_inside + 1):
                for rightprec in xrange(max_inside + 1):
                    for rightsucc in xrange(max_outside + 1):
                        ll = leftprec + leftsucc + rightprec + rightsucc
                        if ll <= 2 and (zero_ok or ll > 0):
                            f = TagEdges(binlen, leftprec, leftsucc, rightprec, rightsucc)
                            tagedges.append(f)

    return tagedges

feature_mapping = {} # string -> id
reverse_mapping = {0: 'LogP'} # id -> string
fclasstable = []   # a collection of feature classes

def _prep_features(names, read_names=True, prep_weights=False):

    # initiliaze implemented feature templates
    
##    r1 = Rule(1)
##    r2 = Rule(2)
##    h = Heavy()
##    w1 = Word(1)
##    w2 = Word(2)
##    rb = RightBranch()

##    wproj = WProj()


##    colenpar = CoLenPar()

##    ngramtree2 = [NGramTree(2, 0)]#, NGramTree(2, 3)]


    ###################
    
    implemented = {'ce-1': wordedges(max_words=1), 'ce-2': wordedges()}
    
##    {'rule-1': [r1], 'rule-2': [r2], \
##                   'heavy': [h], \
##                   'word-1': [w1], 'word-2': [w2], 'word': [w1, w2], \
##                   'rb': [rb], \
##                   'wproj': [wproj], "headtree" : headtree, \
##                   'heads-2' : heads2, 'heads-3': [heads3], 'heads': heads2 + [heads3], \
##                   'wordedges': wordedges(), 'we-1': wordedges(max_words=1), \
##                   'colenpar': [colenpar], \
##                   'ngramtree-2': ngramtree2, \
##                   'bigram' : bigram, 'trigram': trigram, 'headmod': headmod, 'distmod': distmod, \
##                   'collins' : collins, \
##                   'tagedges' : tagedges(), 'insideedges' : tagedges(max_inside=1), 'insideedges2' : tagedges(max_inside=2)
##                   }

    for name in names:
        assert name in implemented, "%s is not implemented" % name
        fclasstable.extend(implemented[name])

##        if read_names:
##            ## f-rule-1 => w-rule-1
##            fname = modelsdir + "/f-" + name
##            wname = modelsdir + "/w-" + name
##            idset = read_features(fname, feature_mapping, reverse_mapping)

##            if prep_weights:
##                read_weights(wname, idset)

##        Vector.feature_mapping = feature_mapping
##        Vector.reverse_mapping = reverse_mapping

    return fclasstable

def prep_features(args, read_names=True):
    return _prep_features(args, read_names=read_names)

#     allfeats = []
#     for names in args:
# ##        print names
#         feats = _prep_features(names.split(), read_names=True)
#         allfeats.append(feats)
#     return allfeats


def pp_id_v((iden, v)):

    return ("%7d    " % iden if v == 1 else "%7d:%-3d" % (iden, v))


def pp_fv(fvector, sentid):
    
    toprint = []
    for Id, v in fvector.items():
        toprint.append((Id, v))
        
    toprint.sort()
    if cross_lines:
        return "\n".join(["#%d\t%s" % (sentid, pp_id_v(id_v)) for id_v in toprint]) + "\n"
    else:
        return "#%d\t%s" % (sentid, "\t".join([pp_id_v(id_v) for id_v in toprint]))


def evaluate(tree, sentence, j=0, logprob=0):        
    fvector = extract(tree, sentence, fclasstable)
    #w = dot_product(fvector, weights) + logprob * probweight
    w=0
    print >> logs, pp_fv(fvector, j)
    return w
    
