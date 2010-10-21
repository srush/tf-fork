# -*- coding: utf-8 -*-
#!/usr/bin/env python

''' class Forest is a collection of nodes, and is responsible for loading/dumping
    the forest.
    The real classes Node and Hyperedge are defined in node_and_hyperedge.py.

    N.B. do not remove sp for an already removed forest.

    Sep 2009: Adapted for ISI translation forest.
'''

# 199 2345
# 1   DT [0-1]    0 
# ...
# 6   NP [0-2]    1 
#       1 4 ||| 0=-5.342
# ...

## N.B. node ID might be 123-45-67 where -x are subcats due to annotations.
    
import sys, os, re
import math
import time
import copy
import fileinput

logs = sys.stderr

from utility import getfile, words_to_chars, quoteattr
from tree import Tree

from svector import Vector   # david's pyx, instead of my fvector
from node_and_hyperedge import Node, Hyperedge

from pattern_matching import PatternMatching

from bleu import Bleu
#import oracle

from utility import desymbol
from rule import Rule, RuleSet

from model import Model

import itertools

import gflags as flags
FLAGS=flags.FLAGS

print_merit = False
cache_same = False

base_weights = Vector("lm1=2 gt_prob=1 plhs=1 text-length=1")

flags.DEFINE_integer("first", None, "first N forests only")
convert_forest = False 
flags.DEFINE_float("lmratio", 0.8, "language model weight multiplier")

class Forest(object):
    ''' a collection of nodes '''

    globalruleid = 0    

    def copy(self):
        '''must be deep!'''
        return copy.deepcopy(self)
        
  #  def size(self):
  #      ''' return (num_nodes, num_edges) pair '''
  #      return len(self.nodes), self.num_edges ##sum([len(node.edges) for node in self.nodes.values()])

    def compute_size(self):
        self.num_edges = sum([len(node.edges) for node in self])

    def size(self):
        ''' return (num_nodes, num_edges) pair '''
        return len(self.nodes), self.num_edges 

    def update_nodes(self, reachable):
        newnodes = {}
        newnodeorder = []
        newedgeorder = []
        edgecounter = 0
        for i, node in enumerate(self):
            if node.iden in reachable:
                node.position_id = i
                newnodes[node.iden] = self.nodes[node.iden]
                newnodeorder.append(node)
                edgecount = node.number_edges(edgecounter, newedgeorder)
        self.nodes = newnodes
        self.nodeorder = newnodeorder
        self.edgeorder = newedgeorder
        self.number_nodes()

    def number_nodes(self):
        newedgeorder = []
        edgecounter = 0
        i = 0
        for i, node in enumerate(self):
            node.position_id = i
            edgecounter = node.number_edges(edgecounter, newedgeorder)
        self.edgeorder = newedgeorder
        self.edgelen = edgecounter
        self.nodelen = i +1
        
    def __init__(self, sent, cased_sent, is_tforest=False, tag=""):
        self.tag = tag
        self.nodes = {}  ## id: node
        self.nodeorder = [] #node

        self._tforest = is_tforest
        
        self.sent = sent
        # a backup of cased, word-based sentence, since sent itself is going to be lowercased and char-based.
        self.cased_sent = cased_sent
        self.len = len(self.sent)
        self.wlen = len(self.cased_sent)

        self.cells = {}   # cells [(2,3)]...
        self.num_edges = 0
        
        #self.num_tfedges = 0

        self.weights = base_weights # baseline

    def __len__(self):
        "sentence length"
        return self.len

    def add_node(self, node):
        self.nodes[node.iden] = node
        self.nodeorder.append(node)
        
        node.forest = self ## important backpointer!
        node.wrd_seq = self.sent[node.span[0]: node.span[1]]

    
    def rehash(self):
        ''' after pruning'''

        for i in xrange(self.len+1):
            for j in xrange(i, self.len+1): ## N.B. null span allowed
                self.cells[(i,j)] = []
        
        self.num_edges = 0
        for node in self:
            self.cells[node.span].append(node)
            self.num_edges += len(node.edges)
        self.number_nodes()

    def clear_bests(self):
        for node in self:
            node.bestres = None

#     def adjust_output(self, (sc, tr, fv)):
#         ## subs[0]: remove TOP level
#         ## no longer turning into negative!
#         return sc, " ".join(tr), fv   #tr.cased_str(self.cased_sent), fv #.subs[0]
    
    def bestparse(self, weights=base_weights, adjust=True, use_min=True):
        """Viterbi-Inside dp on forest.

        """
        self.clear_bests()

        return self.root.bestparse(weights, use_min)

    def sumparse(self, weights=base_weights):
        return self.root.sumparse(weights)

    def prep_kbest(self, weights=base_weights):
        self.bestparse(weights)
        for node in self:
            ## set up klist and kset, but no computation
            node.prepare_kbest()
            
        return self.root.bestres[0]        

    def iterkbest(self, weights, maxk, threshold):
        ''' (lazy) generator '''

        bestscore = self.prep_kbest(weights)
        root = self.root
        for k in xrange(maxk):
            root.lazykbest(k+1)
            if root.fixed or threshold is not None and root.klist[k][0] > bestscore + threshold:
                break            
            else:
                ret = root.klist[k]
                # for psyco
                yield ret
        
    def lazykbest(self, k, weights=base_weights, sentid=0, threshold=None):
        '''return 1-best and k-best times'''

        basetime = time.time()        
        bestscore = self.prep_kbest(weights)
        onebest = time.time()        

        if k > 1:
            # TODO: remove redundant work in k-best
            self.root.lazykbest(k)
        else:
            self.root.klist = [self.root.bestres]
            
        kbest = time.time()

        if threshold is not None:
            for i, (sc, tr, fv) in enumerate(self.root.klist):
                if sc > bestscore + threshold:
                    self.root.klist = self.root.klist[:i]
                    break

        return onebest - basetime, kbest - basetime
                
    @staticmethod
    def load(filename, is_tforest=False, lower=False, sentid=0, first=None, lm=None):
        '''now returns a generator! use load().next() for singleton.
           and read the last line as the gold tree -- TODO: optional!
           and there is an empty line at the end
        '''
        if first is None: # N.B.: must be here, not in the param line (after program initializes)
            first = FLAGS.first
            
        file = getfile(filename)
        line = None
        total_time = 0
        num_sents = 0        
        
        while True:            
            
            start_time = time.time()
            ##'\tThe complicated language in ...\n"
            ## tag is often missing
            line = file.readline()  # emulate seek
            if len(line) == 0:
                break
            try:
                ## strict format, no consecutive breaks
#                 if line is None or line == "\n":
#                     line = "\n"
#                     while line == "\n":
#                         line = file.readline()  # emulate seek
                        
                tag, sent = line.split("\t")   # foreign sentence
                
            except:
                ## no more forests
                yield None
                continue

            num_sents += 1

            # caching the original, word-based, true-case sentence
            sent = sent.split() ## no splitting with " "
            cased_sent = sent [:]            
            if lower:
                sent = [w.lower() for w in sent]   # mark johnson: lowercase all words

            #sent = words_to_chars(sent, encode_back=True)  # split to chars

            ## read in references
            refnum = int(file.readline().strip())
            refs = []
            for i in xrange(refnum):
                refs.append(file.readline().strip())

            ## sizes: number of nodes, number of edges (optional)
            num, nedges = map(int, file.readline().split("\t"))   

            forest = Forest(sent, cased_sent, tag, is_tforest)

            forest.tag = tag

            forest.refs = refs
            forest.bleu = Bleu(refs=refs)  ## initial (empty test) bleu; used repeatedly later
            
            forest.labelspans = {}
            forest.short_edges = {}
            forest.rules = {}

            for i in xrange(1, num+1):

                ## '2\tDT* [0-1]\t1 ||| 1232=2 ...\n'
                ## node-based features here: wordedges, greedyheavy, word(1), [word(2)], ...
                line = file.readline()
                try:
                    keys, fields = line.split(" ||| ")
                except:
                    keys = line
                    fields = ""

                iden, labelspan, size = keys.split("\t") ## iden can be non-ints
                size = int(size)

                fvector = Vector(fields) #
##                remove_blacklist(fvector)
                node = Node(iden, labelspan, size, fvector, sent)
                forest.add_node(node)

                if cache_same:
                    if labelspan in forest.labelspans:
                        node.same = forest.labelspans[labelspan]
                        node.fvector = node.same.fvector
                    else:
                        forest.labelspans[labelspan] = node

                for j in xrange(size):
                    is_oracle = False

                    ## '\t1 ||| 0=8.86276 1=2 3\n'
                    ## N.B.: can't just strip! "\t... ||| ... ||| \n" => 2 fields instead of 3
                    tails, rule, fields = file.readline().strip("\t\n").split(" ||| ")

                    if tails != "" and tails[0] == "*":  #oracle edge
                        is_oracle = True
                        tails = tails[1:]

                    tails = tails.split() ## N.B.: don't split by " "!
                    tailnodes = []
                    lhsstr = [] # 123 "thank" 456

                    lmstr = []
                    lmscore = 0
                    lmlhsstr = []
                    
                    for x in tails:
                        if x[0]=='"': # word
                            word = desymbol(x[1:-1])
                            lhsstr.append(word)  ## desymbol here and only here; ump will call quoteattr
                            
                            if lm is not None:
                                this = lm.word2index(word)
                                lmscore += lm.ngram.wordprob(this, lmstr)
                                lmlhsstr.append(this)
                                lmstr += [this,]
                                
                        else: # variable

                            assert x in forest.nodes, "BAD TOPOL ORDER: node #%s is referred to " % x + \
                                         "(in a hyperedge of node #%s) before being defined" % iden
                            tail = forest.nodes[x]
                            tailnodes.append(tail)
                            lhsstr.append(tail)                            

                            if lm is not None:
                                lmstr = []  # "..." "..." x0 "..."
                                lmlhsstr.append(tail) # sync with lhsstr

                    fvector = Vector(fields)
                    if lm is not None:
                        fvector["lm1"] = lmscore # hack

                    edge = Hyperedge(node, tailnodes, fvector, lhsstr)
                    edge.lmlhsstr = lmlhsstr

                    ## new
                    x = rule.split()
                    edge.ruleid = int(x[0])
                    if len(x) > 1:
                        edge.rule = Rule.parse(" ".join(x[1:]) + " ### " + fields)
                        forest.rules[edge.ruleid] = edge.rule #" ".join(x[1:]) #, None)
                    else:
                        edge.rule = forest.rules[edge.ruleid] # cahced rule

                    node.add_edge(edge)
                    if is_oracle:
                        node.oracle_edge = edge
                    
                if node.sp_terminal():
                    node.word = node.edges[0].subs[0].word

            ## splitted nodes 12-3-4 => (12, 3, 4)
            tmp = sorted([(map(int, x.iden.split("-")), x) for x in forest.nodeorder])   
            forest.nodeorder = [x for (_, x) in tmp]

            forest.rehash()
            sentid += 1
            
##            print >> logs, "sent #%d %s, %d words, %d nodes, %d edges, loaded in %.2lf secs" \
##                  % (sentid, forest.tag, forest.len, num, forest.num_edges, time.time() - basetime)

            forest.root = node
            node.set_root(True)
            line = file.readline()

            if line is not None and line.strip() != "":
                if line[0] == "(":
                    forest.goldtree = Tree.parse(line.strip(), trunc=True, lower=False)
                    line = file.readline()
            else:
                line = None

            forest.number_nodes()
            #print forest.root.position_id
          

            total_time += time.time() - start_time

            if num_sents % 100 == 0:
                print >> logs, "... %d sents loaded (%.2lf secs per sent) ..." \
                      % (num_sents, total_time/num_sents)

            forest.subtree() #compute the subtree string for each node

            yield forest

            if first is not None and num_sents >= first:
                break                

        # better check here instead of zero-division exception
        if num_sents == 0:
            print >> logs, "NO FORESTS FOUND!!! (empty input file?)"
            sys.exit(1)            
#            yield None # new: don't halt -- WHY?
        
        Forest.load_time = total_time
        print >> logs, "%d forests loaded in %.2lf secs (avg %.2lf per sent)" \
              % (num_sents, total_time, total_time/(num_sents+0.001))

    @staticmethod
    def loadall(filename, is_tforest=False):
        return list(Forest.load(filename, is_tforest))

    def subtree(self):
        for node in self:
            node.psubtree()

    def dump(self, out=sys.stdout):
        '''output to stdout'''
        # wsj_00.00       No , it was n't Black Monday .
        # 199
        # 1    DT [0-1]    0 ||| 12321=... 46456=...
        # ...
        # 6    NP [0-2]    1 ||| 21213=... 7987=...
        #     1 4 ||| 0=-5.342
        # ...
        
        if type(out) is str:
            out = open(out, "wt")

        # CAUTION! use original cased_sent!
        print >> out, "%s\t%s" % (self.tag, " ".join(self.cased_sent))
        print >> out, len(self.refs)
        for ref in self.refs:
            print >> out, ref
        
        print >> out, "%d\t%d" % self.size()  # nums of nodes and edges
        rulecache = set()
        for node in self:

            oracle_edge = node.oracle_edge if hasattr(node, "oracle_edge") else None
            
            print >> out, "%s\t%d |||" % (node.labelspan(separator="\t"), len(node.edges)),
            if hasattr(node, "same"):
                print >> out, " "
            else:
                print >> out, node.fvector
                
            ##print >> out, "||| %.4lf" % node.merit if print_merit else ""

            for edge in node.edges:

                is_oracle = "*" if (edge is oracle_edge) else ""

                # TODO: merge
                if hasattr(edge.rule, "ruleid"):
                    edge.ruleid = edge.rule.ruleid

                ## caution: pruning might change caching, so make sure rule is defined in the output forest
                if edge.ruleid in rulecache:
                    rule_print = str(edge.ruleid)
                else:
                    rule_print = "%s %s" % \
                                 (edge.ruleid, edge.rule.bp_print(node.subtree))
                    rulecache.add(edge.ruleid)
                wordnum = sum([1 if type(x) is str else 0 for x in edge.lhsstr])
                tailstr = " ".join(['"%s"' % x if type(x) is str else x.iden for x in edge.lhsstr])

                if convert_forest: # convert forest
                    edge.fvector["rule-num"] = 1
                    edge.fvector["text-length"] = wordnum
                    
                print >> out, "\t%s%s ||| %s ||| %s" \
                            % (is_oracle, tailstr, rule_print, edge.fvector)
                     
        print >> out  ## last blank line

    def __iter__(self):
        for node in self.nodeorder:
            yield node

    def reverse(self):
        for i in range(len(self.nodeorder)):
            ret = self.nodeorder[-(i+1)]            
            yield ret

    ## from oracle.py
    def recover_oracle(self):
        '''oracle is already stored implicitly in the forest
        returns best_score, best_parseval, best_tree, edgelist
        '''
        edgelist = self.root.get_oracle_edgelist()
        fv = Hyperedge.deriv2fvector(edgelist)
        tr = Hyperedge.deriv2tree(edgelist)
        bleu_p1 = self.bleu.rescore(tr)
        return bleu_p1, tr, fv, edgelist


    def compute_oracle(self, weights, model_weight=0, bleu_weight=1, store_oracle=False):
        '''idea: annotate each hyperedge with oracle state. and compute logBLEU for
        each node. NOTE that BLEU is highly non-decomposable, so this dynamic programming
        is a very crude approximation. alternatively, we can use Tromble et al decomposable
        BLEU (so that it becomes viterbi deriv), but then we need to tune the coefficients.'''


        basetime = time.time()
        bleu, hyp, fv, edgelist = self.root.compute_oracle(weights, self.bleu,
                                                           self.len, self.wlen,
                                                           model_weight, bleu_weight)
        bleu = self.bleu.rescore(hyp) ## for safety, 755 bug
        
        if store_oracle:
            for edge in edgelist:
                edge.head.oracle_edge = edge
                
        return bleu, hyp, fv, edgelist

###
def output_avg_stats():

    print >> logs,  "overall %d sents 1-best deriv bleu = %s score = %.4f\t1-best %.3f secs  %d-best %.3f secs" \
          % (i, onebestbleus.score_ratio_str(), onebestscores/i, total1besttime/i, FLAGS.k, totalkbesttime/i)

    if FLAGS.oracle:
        print >> logs,  "overall %d sents my    oracle bleu = %s score = %.4lf\t  time %.3f secs" \
              % (i, myoraclebleus.score_ratio_str(), myscores/i, totaloracletime/i)

def filter_input(filename):
    for line in open(filename):
        #AD("60.3") -> "60.3" ### 0:0 fields: count=1 id=6
        lhs, other = line.strip().split(' -> ', 1)
        #count1 = float(other.split('count1=', 1)[1].split()[0])
        yield (lhs, line)



if __name__ == "__main__":

    flags.DEFINE_string("ruleset", None, "translation rule set (parse => trans)", short_name="r")
##    flags.DEFINE_boolean("trans", False, "translation forest instead of parse forest", short_name="t")
    flags.DEFINE_string("phrase", None, "bilingual phrase from Moses")
    flags.DEFINE_boolean("oracle", False, "compute oracles")
    flags.DEFINE_integer("kbest", 1, "kbest", short_name="k")
    flags.DEFINE_boolean("dump", False, "dump forest (to stdout)")    
    flags.DEFINE_boolean("infinite", False, "infinite-kbest")    
    flags.DEFINE_float("threshold", None, "threshold/margin")

    flags.DEFINE_string("rulefilter", None, "filter ruleset")
    flags.DEFINE_integer("max_height", 3, "maximum height of lhs for pattern-matching")


    flags.DEFINE_integer("example_limit", 1e10, "number of examples to use")
    
    flags.DEFINE_float("hope", 0, "hope weight")
    flags.DEFINE_boolean("mert", True, "output mert-friendly info (<hyp><cost)")

    from ngram import Ngram # defines --lm and --order    

    argv = FLAGS(sys.argv)

    weights = Model.cmdline_model()
    lm = Ngram.cmdline_ngram() # if FLAGS.lm is None then returns None
    if lm:
        weights["lm1"] = weights["lm"] * FLAGS.lmratio

    reffiles = [open(f) for f in argv[1:]]

    convert_forest = ((FLAGS.ruleset is not None) or (FLAGS.rulefilter is not None) )
  
    if FLAGS.ruleset is not None:
        ruleset = RuleSet(FLAGS.ruleset)
        
        if FLAGS.phrase is not None:
            ruleset.add_bp(FLAGS.phrase)

        Forest.globalruleid = ruleset.rule_num()
    
    davidoraclebleus = Bleu()

    myoraclebleus = Bleu()
    myfearbleus = Bleu()
    davidscores = 0
    myscores = 0
    myfearscores = 0
    onebestscores = 0
    onebestbleus = Bleu()
    filtered_ruleset = {}
    totalconvtime = 0
    total1besttime = 0
    totalkbesttime = 0
    totaloracletime = 0
    totalfiltertime = 0
    
    if FLAGS.rulefilter is not None:
        all_lhss = set()
        
    for i, forest in enumerate(Forest.load("-", is_tforest=(not convert_forest), lm=lm), 1):
        if i > FLAGS.example_limit: break
        
        if not convert_forest:  # translation forest
            if not FLAGS.infinite:
                if FLAGS.k is None:
                    FLAGS.k = 1

                onebesttime, kbesttime = \
                             forest.lazykbest(FLAGS.k, weights=weights, sentid=forest.tag, threshold=FLAGS.threshold)
                
                print >> logs, "sent #%-4d\t1-best time %.3f secs, total %d-best time %.3f secs" % \
                      (i, onebesttime, FLAGS.k, kbesttime)

                total1besttime += onebesttime
                totalkbesttime += kbesttime

                if FLAGS.mert:
                    print >> logs, '<sent No="%d">' % i
                    print >> logs, "<Chinese>%s</Chinese>" % " ".join(forest.cased_sent)
                
##                forest.root.print_derivation()

                for k, res in enumerate(forest.root.klist):
                    score, hyp, fv = res
                    hyp = (hyp)
                    hyp_bleu = forest.bleu.rescore(hyp)
                    
                    print >> logs, "k=%d\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\t%s" % \
                          (k+1, score, hyp_bleu, forest.bleu.ratio(), fv)
                    #print hyp # to stdout

                    #for MERT output
                    if FLAGS.mert:
                        print >> logs, "<score>%.3lf</score>" % score
                        print >> logs, "<hyp>%s</hyp>" % hyp
                        print >> logs, "<cost>%s</cost>" % fv
                        
                    if k == 0: #1-best
                        print hyp # to stdout
                        onebestscores += score
                        onebestbleus += (hyp, forest.refs)#forest.bleu.copy()

                if FLAGS.oracle:
                    basetime = time.time()

                    bleu, hyp, fv, edgelist = forest.compute_oracle(weights, FLAGS.hope, 1)

                    oracletime = time.time() - basetime
                    totaloracletime += oracletime
                    
                    bleu = forest.bleu.rescore(hyp)
                    mscore = weights.dot(fv)
                    print >> logs, "moracle\tscore=%.4lf\tbleu+1=%.4lf\tlenratio=%.2lf\t%s" % \
                          (mscore, forest.bleu.fscore(), forest.bleu.ratio(), fv)
                    print >> logs, hyp

                    myoraclebleus += forest.bleu.copy()
                    myscores += mscore

                # for MERT output
                if FLAGS.mert:
                    print >> logs, "</sent>"
                
            else:
                if FLAGS.k is None:
                    FLAGS.k = 100000 ## inf
                for res in forest.iterkbest(FLAGS.k, threshold=FLAGS.threshold):
                    print >> logs,  "%.4lf\n%s" % (forest.adjust_output(res)[:2])

            if i % 10 == 0:
                output_avg_stats()

        elif FLAGS.rulefilter is not None:
            # filter rule set
            stime = time.time()
            # print >> logs, "start rule filter ..." 
            pm = PatternMatching(forest, {}, '', FLAGS.max_height, True, FLAGS.phrase)
            all_lhss |= pm.convert()
            etime = time.time()
            totalfiltertime += (etime - stime)

        else:
            # convert pforest to tforest by pattern matching 
            stime = time.time()
            # default fields
            deffields = "gt_prob=-50 proot=-50 prhs=-20 plhs=-20 lexpef=-10 lexpfe=-10 count1=0 count2_3=0 count4=0 prhs_var=-20 plhs_var=-20 ghkm_num=0 bp_num=0 def_num=1"
            # inside replace
            pm = PatternMatching(forest, ruleset, deffields, FLAGS.max_height, False, FLAGS.phrase)
            forest = pm.convert()
            forest.compute_size()
            forest.refs = [f.readline().strip() for f in reffiles]
            forest.dump()

            etime = time.time()
            print >> logs, "sent: %s, len: %d, nodes: %d, edges: %d, \tconvert time: %.2lf" % \
                  (forest.tag, len(forest), forest.size()[0], forest.size()[1], etime - stime)
            totalconvtime += (etime - stime)
            
    if FLAGS.ruleset:
        print >> logs, "Total converting time: %.2lf" % totalconvtime

    elif FLAGS.rulefilter is not None:
        print >> logs, "Start output filtered rules ..."
            
        for key, val in itertools.groupby(filter_input(FLAGS.rulefilter), lambda x: x[0]):
            if key in all_lhss:
                for (_, line) in val:
                    print line.strip()  

        print >> logs, "Total rulefilting time: %.2lf" % totalfiltertime

    if not convert_forest:  # translation forest
        output_avg_stats()
