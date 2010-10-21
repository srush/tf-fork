#!/usr/bin/env python

from collections import defaultdict
import sys
import time
logs = sys.stderr

import gflags as flags
FLAGS=flags.FLAGS

from utility import quoteattr

class Rule(object):

    __slots__ = "lhs", "rhs", "fields", "ruleid", "_tree_size"

    def __init__(self, lhs, rhs, fields):
        self.lhs = lhs
        self.rhs = rhs
        self.fields = fields
        # TODO: -LRB--
        self._tree_size = self.lhs.count("(") - self.lhs.count("\"(") # LRB


    @staticmethod
    def parse(line):
        '''S(NP("...") VP) -> x1 "..." x0 ### gt_prob=-5 text-lenght=0 plhs=-2'''
        try:
            rule, fields = line.split(" ### ")
            lhs, rhs = rule.strip().split(" -> ")
            rhs = rhs.split()
            return Rule(lhs, rhs, fields.strip())
        except:
            print >> logs, "BAD RULE: %s" % line.strip()
            return None

    def __str__(self):
        return "%s ### %s" % (repr(self), self.fields)

    def __repr__(self):
        return "%s -> %s" % (self.lhs, \
                             " ".join('"%s"' % s[1:-1] if s[0] == '"' else s \
                                    for s in self.rhs))
                     #        " ".join(quoteattr(s[1:-1]) if s[0] == '"' else s \
                      #                for s in self.rhs))

    def bp_print(self, subtree):
        # ATTENTION:
        if not self.is_bp():
            subtree = self.lhs
        return "%s -> %s" % (subtree, \
                             " ".join('"%s"' % s[1:-1] if s[0] == '"' else s \
                                    for s in self.rhs))

    def tree_size(self):
        '''number of non-variable nodes in lhs tree'''
        return self._tree_size

    def is_bp(self):
        '''am i bilingual phrase rule?'''
        return self.lhs.count("(") == 0

    def __hash__(self):
       return hash((self.lhs, tuple(self.rhs)))

    def __eq__(self, other):
       return self.lhs==other.lhs and self.rhs==other.rhs

    
class RuleSet(defaultdict):
    ''' map from lhs to a list of Rule '''

    def __init__(self, rulefilename):
        '''read in rules from a file'''
        defaultdict.__init__(self, list) # N.B. superclass
        self.ruleid = 0
        print >> logs, "reading rules from %s ..." % rulefilename

        ## YY new switch
        if isinstance(rulefilename, str):
           self.add_rulefromfile(rulefilename)
        else: # list/iterator   
           self.from_strlist(rulefilename)
        
    # YYnew
    def from_strlist(self, lst):
        otime = time.time()
        bad = 0
        for i, line in enumerate(lst, 1):
            rule = Rule.parse(line)
            if rule is not None:
                self.add_rule(rule)
            else:
                bad += 1
            if i % 100000 == 0:
                print >> logs, "%d rules read (%d BAD)..." % (i, bad) 
        print >> logs, "\ntotal %d rules (%d BAD) read in %d secs" % (i, bad, time.time() - otime)

    #YYnote: this can change to just use from_strlist
    def add_rulefromfile(self, rulefilename):
        otime = time.time()
        bad = 0
        for i, line in enumerate(open(rulefilename), 1):
            rule = Rule.parse(line)
            if rule is not None:
                self.add_rule(rule)
            else:
                bad += 1
            if i % 100000 == 0:
                print >> logs, "%d rules read (%d BAD)..." % (i, bad) 
            
        print >> logs, "\ntotal %d rules (%d BAD) read in %d secs" % (i, bad, time.time() - otime)

    def add_bp(self, bpfile):
        print >> logs, 'reading bp rules from %s ...' % bpfile
        self.add_rulefromfile(bpfile)
        
    def add_rule(self, rule):
        self[rule.lhs].append(rule)
        rule.ruleid = self.ruleid
        self.ruleid += 1
        
    def rule_num(self):
        return self.ruleid

if __name__ == "__main__":
    #import optparse
    #optparser = optparse.OptionParser(usage="usage: rule.py 1>filtered.rules 2>bad.rules")
    flags.DEFINE_integer("max_ce_ratio", 4, "maximum ce tatio of garbage rules")

    print >> logs, "start to filting rule set ..."
    stime = time.time()
    
    bad_ratio = 0    #bad ratio
    bad_noalign = 0   # NO algnments
    bad_null = 0     # rhs == NULL

    # filtered rule set
    # filteredrs = defaultdict(list)
    for i, line in enumerate(sys.stdin, 1):
        rule = Rule.parse(line)
        if rule is not None:
            ratioce = float(len(rule.lhs.split()))/float(len(rule.rhs))
            #ratioec = float(len(rule.rhs))/float(len(rule.lhs.split()))
            if ratioce > 4:
                bad_ratio += 1
                print >> logs, "Bad Ratio: %s" % line.strip()
            elif ":" not in rule.fields:
                bad_noalign += 1
                print >> logs, "No alginment: %s" % line.strip()
            else:
                # filteredrs[rule.lhs].append(rule)
                print line.strip()
        else:
            bad_null += 1

#    for (lhs, rules) in filteredrs.iteritems():
#        for rule in rules:
#            print "%s" % str(rule)

    etime = time.time()

    print >> logs, "\ntotal number of rules: %d" % i
    print >> logs, "\t %d rhs=null; %d bad ratio; %d no alignment" % (bad_null, bad_ratio, bad_noalign)
    print >> logs, "\t %.2lf rhs=null; %.2lf bad ratio; %.2lf no alignment" %\
          (float(bad_null)/float(i), float(bad_ratio)/float(i), float(bad_noalign)/float(i))
