from itertools import *

class RuleTree(object): #{{{
   __slots__= 'label','subs'

   def __init__(self, label, subs):
      self.label = label
      self.subs  = subs
      if not label: print "creating rule with empty label!",self

   def __repr__(self): return "<Rule/lhs:%s>" % self.pp()

   def pp(self):
      if not self.subs: return "%s" % self.label
      return "%s(%s)" % (self.label, " ".join([x.pp() for x in self.subs]))

   @classmethod
   def from_rule(cls, rule, no_words=False):
      return cls.from_lhs_string(rule.lhs, no_words=no_words)

   #{{{ ghkm rule string lhs parsing
   @staticmethod
   def _consume_subs(string, start=0, should_close=False, no_words=False):
      """
      start should be positioned AFTER the opening "("
      no_words: don't include word tokens in the resulting tree
      """
      res = []
      while string[start]!=")":
         brack = string.find("(",start)
         space = string.find(" ",start)
         if brack>-1 and brack < space: # NP(x y z
            label = string[start:brack]
            subs, newstart = RuleTree._consume_subs(string,brack+1, should_close=True, no_words=no_words)
            start = newstart
            res.append(RuleTree(label, subs))
         elif space>-1: # X
            label = string[start:space]
            start = space+1
            if not label: # final space before ")"
               continue
            if no_words and label[0]=='"':
               #assert(label[0]==label[-1])
               continue
            res.append(RuleTree(label, []))
         else:
            assert(False),"should not get here.."
         # I want "should_close" so I can identify errors.  
         # otherwise can just change the loop condition to include start<len(string)
         if (not should_close) and start>=len(string): break
      return res, start+1
   #}}} 

   @classmethod
   def from_lhs_string(cls, lhs, no_words=False):
      # corner case with just one atom: NP
      if lhs.find("(") == -1: return RuleTree(lhs,[])
      # makes thing easier for the parser
      lhs = lhs.replace(")"," )")
      t, e = RuleTree._consume_subs(lhs, no_words=no_words)
      assert(len(t)==1)
      return t[0]

   def match(self,  eforest_node, acc=None):
      '''
      NOTE: only tested when eforest is just one tree 
      acc: will accumulate the nodes that are combined by this rule (leaf nodes) (does not make sense for no-match)
         acc contain pairs: [(node,is_terminal:boolean),...]
            where is_terminal==False means "variable"
      '''
      if acc is None: acc=[]
      #print "matching:",self.label, eforest_node.label, eforest_node.span
      if self.label != eforest_node.label: return False
      #if (not self.subs) or (len(self.subs)==1 and self.subs[0].label[0]=='"'):
      #   acc.append((eforest_node,False if not self.subs else self.subs[0].label[0]=='"'))
      #   return True
      if (not self.subs):
         acc.append((eforest_node,False))
         return True
      if (len(self.subs)==1 and self.subs[0].label[0]=='"'):
         acc.append((eforest_node,self.subs[0].label[0]=='"'))
         return True
      for edge in eforest_node.edges:
         esubs = edge.subs
         if len(self.subs)==len(esubs) and all( (s.match(es,acc) for s,es in izip(self.subs,esubs)) ):
            return True
      # no match for any edge: return False
      return False
#}}}

# it can wrap a TreeRule and provide a non-recursive .match(forest_node) function
# thought it'd be faster. it is actually a bit slower..
# update: slower when match, faster when fail.. so that's good.
#     also takes less memory to store it
# TODO: maybe can be made faster by storing two lists, one for nodes and one for counts,
#       and then iterating over zip(label,nchilds) instead of label in iter(rep) and nchilds=rep.next()
class NonrecTreeMatcher:
   __slots__ = 'rep'
   def __init__(self, rep):
      self.rep = rep

   def __repr__(self): return "<NonrecTreeMatcher:%s>" % self.pp()

   def pp(self):
      return ' '.join(map(str,self.rep))

   @staticmethod
   def from_ruletree(rt):
      rep = []
      stack = [rt]
      while stack:
         r = stack.pop()
         if r.label[0]=='"':
            rep.append(None)
         else:
            rep.append(r.label)
         if not r.subs:
            if r.label[0]!='"':
               rep.append(0)
         else:
            rep.append(len(r.subs))
            stack.extend(reversed(r.subs))
      return NonrecTreeMatcher(rep)
   
   def match(self, eforest_node, acc=[]):#,D=False):
      pnode=None  # keeps the tree-node for terminals
      rep = iter(self.rep)
      tstack = [eforest_node]
      for label in rep:
         node = tstack.pop()
         if node is None: # a lexical node in the tree
            if label is not None: return False
            acc.append((pnode,True))
            continue

         if label != node.label: 
            return False

         nchilds = rep.next()
         if nchilds==0:
            acc.append((node,node is None))
            continue

         edges = node.edges
         assert(len(edges)<=1),len(edges)
         if len(edges)==0: 
            assert(node.is_terminal())
            pnode=node
            tstack.append(None)  # None instead of _DummyNode
            continue

         subs = edges[0].subs
         if len(subs) != nchilds:
            return False
         tstack.extend(reversed(subs))
      return True
         


if __name__=='__main__':
   rules = ['''X''', '''VP(x0:VBZ x1:NP-C)''', '''S(''(""") NP-C(NPB(DT("the") x0:NML x1:NN)) x2:VP x3:.)''', '''A(B(C D(E) F) G(T))''']
   print RuleTree.from_lhs_string(rules[0],no_words=True)
   print RuleTree.from_lhs_string(rules[1],no_words=True)
   print RuleTree.from_lhs_string(rules[2],no_words=True)
   print RuleTree.from_lhs_string(rules[2],no_words=False)

   for r in rules:
      print RuleTree.from_lhs_string(r,no_words=False)
      NonrecTreeMatcher.from_ruletree(RuleTree.from_lhs_string(r,no_words=False))
