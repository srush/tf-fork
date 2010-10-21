#!/usr/bin/env python

import sys
logs = sys.stderr

class XRSTree(object):

    __slots__ = "label", "subs"
    
    def __init__(self, label, subs=[]):
        self.label = label
        self.subs = subs

    def cfgrules(self):
        s = []
        if self.subs != []:
            s = ["%s->%s" % (self.label, "_".join([x.label for x in self.subs]))]                                                
            for sub in self.subs:
                s += sub.cfgrules()
                    
        return s

    @staticmethod
    def parse(s):
        _, tree = XRSTree._parse(s + " ", 0)
        return tree

    @staticmethod
    def _parse(s, index):     
        subs = [] #; print "@", s[index:index+4]        
        if s[index] == '"': ## terminal
            i = s.find('"', index+1)
            if i == index + 1: # """
                    i += 1
            label = s[index+1:i]
        elif s[index] == "x": ## variable leaf                
            i = min(s.find(')', index+1), s.find(' ', index+1))
            label = s[s.find(':', index)+1:i]
            if s[i] == ')':
                i -= 1
        else: ## nonterminal
            i = s.find('(', index+1)
            label = s[index:i]
            i += 1
            while s[i] != ')':
                i, sub = XRSTree._parse(s, i)
                subs.append(sub); #         print "-@", s[i:i+4]         
                if s[i] == " ":
                    i += 1                

        return i+1, XRSTree(label, subs)
                

if __name__ == "__main__":

        print XRSTree.parse('@NPB-2(NNS-0("projects") x0:PRP ASDF(x1:PRP))').cfgrules()
