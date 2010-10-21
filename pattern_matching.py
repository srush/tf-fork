#!/usr/bin/env python

import sys
from collections import defaultdict
import time

from node_and_hyperedge import Node, Hyperedge
from rule import Rule, RuleSet

from svector import Vector

logs = sys.stderr

class PatternMatching(object):

    def __init__(self, forest, ruleset, deffields, max_height, rule_filter, use_bp):
        self.forest = forest
        self.ruleset = ruleset
        self.max_height = max_height
        self.deffields = deffields
        self.filter = rule_filter
        self.use_bp = True if use_bp is not None else False

        if self.filter:
            self.all_lhss = set()

        # store all the descendants of a node {node, (descendant nodes)}
        self.descendants = defaultdict(lambda : set())
    
    def lexfrag(self, node):
        '''add lexical frag (lhs, rsh, height) = (PU("."), [], 1)'''
        return ('%s("%s")' % (node.label, node.word), [], 1)
    
    def deffrag(self, node):
        '''# add default frag (lhs, rsh, height) = (PU, [node], 0)'''
        return (node.label, [node], 0)
    
    def convert(self):
        '''convert parse forest into translation forest'''
        forest = self.forest
        
        for node in forest:

            if node.is_terminal():  # a leaf node
                #append the default frag (lhs, rhs, height) = (PU, node, 0)
                node.frags.append(self.deffrag(node))
                #appedn the lexical frag (lhs, rhs, height) = (PU("."), [], 1)
                frag = self.lexfrag(node)
                node.frags.append(frag)

                # NOTICE: Clear the original prasing hyperedges
                node.edges = []

                if self.filter:  # only for rule filter, don't have to generate translation forest
                    self.all_lhss.add(frag[0])
                else:
                    # add lexical translation hyperedges
                    self.add_lex_th(frag[0], node, False)
                    if self.use_bp:
                        # add phrase
                        self.add_lex_th('"%s"' % node.word, node, True)
                
            else:  # it's a non-terminal node
                #append the default frag (lhs, rhs, height) = (PU, node, 0)
                node.frags.append(self.deffrag(node))
                # add non-terminal translation hyperedges
                self.add_nonter_th(node)

                if self.use_bp:
                    # add phrase
                    self.add_lex_th('"%s"' % node.surface, node, True)

        if self.filter:
            return self.all_lhss
        else:
            self.remove_unreach()      
            return forest

    def remove_unreach(self):
        rootid = self.forest.root.iden
        descendants = self.descendants

        #reachable node set
        reachable = set([rootid])
        expendset = set([rootid])
        expended = set()
        # print "start to expend ..."
        while len(expendset) > 0:
            exp = expendset.pop()
            expended.add(exp)
            curdes = descendants[exp]
            reachable |= curdes
            expendset |= curdes
            expendset -= expended
            
        #print >> logs, "reachable set"
        #print >> logs, reachable
        self.forest.update_nodes(reachable)    

    @staticmethod
    def combinetwofrags(basefrag, varfrag, id, lastchild, max_height):
        '''combine two frags'''
        blhs, brhs, bheight = basefrag
        vlhs, vrhs, vheight = varfrag
        # compute the frag height
        height = bheight if bheight > (vheight + 1) else (vheight + 1)
        if height > max_height:
            return None
        
        lhs = "%s %s" % (blhs, vlhs) if id>0 else "%s%s" % (blhs, vlhs)
        if id == lastchild:
            lhs += ")"
        
        rhs = []
        rhs.extend(brhs)
        rhs.extend(vrhs)
       
        return (lhs, rhs, height)

    def add_lex_th(self, lhs, node, isBP):
        ''' add lexical translation rules '''
        ruleset = self.ruleset
        if lhs in ruleset:
            rules = ruleset[lhs]
            
            # add all translation hyperedges
            for rule in rules:
                newrhs = [x[1:-1] for x in rule.rhs]
                tfedge = Hyperedge(node, [], Vector(rule.fields), newrhs)
                tfedge.rule = rule
                node.edges.append(tfedge)
                        
        elif not isBP: # add a default translation hyperedge (monotonic)
            defword = '"%s"' % node.word
            rule = Rule(lhs, [defword], self.deffields)
            tfedge = Hyperedge(node, [], Vector(self.deffields), [defword[1:-1]])
            tfedge.rule = rule
            ruleset.add_rule(rule)
            node.edges.append(tfedge)
        
    def add_nonter_th(self, node):
        ''' add translation hyperedges to non-terminal node '''
        ruleset = self.ruleset
        tfedges = []
        for edge in node.edges:
            # enumerate all the possible frags
            basefrags = [("%s(" % node.label, [], 1)]
            lastchild = len(edge.subs) - 1
            if len(edge.subs) >= 5: # this guy has too many children! it cannot be matched!
                deflhs = "%s(%s)" % (node.label, " ".join(sub.label for sub in edge.subs))
                defrhs = edge.subs
                defheight = 1
                basefrags = [(deflhs, defrhs, defheight)]
            else:
                for (id, sub) in enumerate(edge.subs):
                    oldfrags = basefrags
                    # cross-product
                    basefrags = [PatternMatching.combinetwofrags(oldfrag, frag, id, lastchild, self.max_height) \
                                 for oldfrag in oldfrags for frag in sub.frags]

            # for each frag add translation hyperedges
            for extfrag in basefrags:
                (extlhs, extrhs, extheight) = extfrag
                # add frags
                if extheight <= self.max_height - 1:
                    node.frags.append(extfrag)

                if self.filter:
                    self.all_lhss.add(extlhs)
                else:
                    # add translation hyperedges
                    if extlhs in ruleset:
                        for des in extrhs:
                            self.descendants[node.iden].add(des.iden) #unit(set(extrhs))
                        #print self.descendants[node.iden]
                    
                        rules = ruleset[extlhs]
                
                        # add all translation hyperedges
                        for rule in rules:
                            rhsstr = [x[1:-1] if x[0]=='"' \
                                      else extrhs[int(x.split('x')[1])] \
                                      for x in rule.rhs]
                            tfedge = Hyperedge(node, extrhs,\
                                     Vector(rule.fields), rhsstr)
                            tfedge.rule = rule
                            tfedges.append(tfedge)
            
            if (not self.filter) and (len(tfedges) == 0):  # no translation hyperedge
                for des in edge.subs:
                    self.descendants[node.iden].add(des.iden) #unit(set(edge.subs))
                # add a default translation hyperedge
                deflhs = "%s(%s)" % (node.label, " ".join(sub.label for sub in edge.subs))
                defrhs = ["x%d" % i for i, _ in enumerate(edge.subs)] # N.B.: do not supply str
                defrule = Rule(deflhs, defrhs, self.deffields)
                tfedge = Hyperedge(node, edge.subs,\
                                   Vector(self.deffields), edge.subs)
                tfedge.rule = defrule
                ruleset.add_rule(defrule)
                tfedges.append(tfedge)

        if not self.filter:
            # inside replace
            node.edges = tfedges
