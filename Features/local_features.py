# -*- coding: utf-8 -*-

"""
Local features for GHKM rules
"""

import sys, itertools
from itertools import *
from features import *
from ruletree import RuleTree
from clusters import Clusters

import inspect, local_features, unittest
sys.path.append('..')
from rule import Rule
from tree import Tree
from node_and_hyperedge import Node, Hyperedge
from svector import Vector
import forest

def is_lex(item):
  return item[0] == '"'

def get_var(item):
  assert item[0] == 'x'
  return int(item[1:])

def single_count(ls):
  return [ (l, 1.0) for l in ls]

#TEST RULE


test_rule = \
  "VP(VV VP(VV NP NP(\"¬ÅÊîÅøÅÂ¬Å∫¬ú\"))) -> x1 \"that\" x0 \"considered\" x2 ### gt_prob=-15.000835 plhs=-6.04973345523 rule-num=1 text-length=2 0:1=0 "

#5 "that" 4 "considered" 15 ||| 36 VP(VV VP(VV NP)) -> x1 "that" x0 "considered" x2 ||| gt_prob=-15.0008 plhs=-6.04973 text-length=2 rule-num=1
test_sent = "¬ÅÊîÅøÅÂ¬Å∫¬ú ¬Å‰Åª¬é¬ÅÊùÅ• ¬Å‰Å∏¬ç¬Å‰Åº¬ö ¬ÅÂèÇ¬Å‰Å∏¬é ¬ÅÂÅØ¬ÅªÅÊ¬âÅæ ¬ÅÈòÅ¥ÅË¬Å∞¬ã ¬ÅÁöÑ ¬ÅËÅ°¬å¬Å‰Å∏¬Å∫ . \"".split()

# test_sent = []

# 17	 VP [2-8]	125 ||| 
test_node = Node("17", "VP [2-8]" ,125,  Vector(), test_sent)

test_edge = Hyperedge(test_node, [], Vector("gt_prob=-15.000835 plhs=-6.04973345523 rule-num=1 text-length=2 0:1=1"),"")


class LocalNodeContext(object):
  "Context of just the (node, sent)"
  def __init__(self, node, sentence):
    self.node = node
    self.sent = sentence

  @staticmethod
  def test_produce_context():
    "For testing, produce a static context"

    return LocalNodeContext(test_node, test_sent)
    

class LocalContext(object):
  "Full Local context (node,edge, rule, sent) "
  SOURCE = 1
  TARGET = 2

  def __init__(self, top_node, edge, rule, sentence):
    self.local_node_cxt = LocalNodeContext(top_node, sentence)

    self.node = top_node
    self.edge = edge
    self.rule = rule
    self.sent = sentence
    
    self.fields = dict((a, b) for a,b in edge.fvector.iteritems())

    self.cluster_rhs = self.rule.rhs

    self.treelet = RuleTree.from_lhs_string(self.rule.lhs)
    self.sent = sentence

    self.clustering = False

  def set_cluster_level(self, target_data, bits):
    self.clustering = True
    self.cluster_level = bits

    self.cluster_rhs = []
    for item in self.rule.rhs:
      if is_lex(item):
        bitstring = target_data.lex_to_cluster(item, bits)
        self.cluster_rhs.append(bitstring)
      else:
        self.cluster_rhs.append(item)
    
  def get_fringe_lhs(self):
    "get the fringe of the treelet"
    return self._fringe(self.treelet)


  def get_fixed_rhs(self):
    "rhs of rule with variables replaced by chinese NT"
    return self._replace_rhs(self.treelet, self.cluster_rhs)

  @staticmethod
  def _replace_rhs(tree,rhs):
    lhs_fringe = LocalContext._fringe_no_lex(tree)
    rhs_fringe =  [f for f in rhs if not is_lex(f)] 
    assert len(lhs_fringe) == len(rhs_fringe)

    ret = []
    for item in rhs:
      if is_lex(item):
        ret.append(item)
      else:
        pos = get_var(item)
        ret.append(lhs_fringe[pos])
    return ret

  

  @staticmethod
  def _fringe_no_lex(tree):
    return [f for f in LocalContext._fringe(tree) if not is_lex(f)] 

    
  @staticmethod
  def _fringe(tree):
    if not tree.subs:
      return [tree.label]
    return sum([LocalContext._fringe(child) for child in tree.subs], [])


  # TESTING

  @staticmethod
  def test_produce_context():
    "For testing, produce a static context"
    rule = Rule.parse(test_rule)
    return LocalContext(test_node, test_edge, rule, test_sent)


class TargetDataManager(object):


  def __init__(self, cluster_file = "/home/nlg-02/ar_009/paths"):
    self.clusters = Clusters.read_clusters(cluster_file)
    #self.cache = {}
    
  def lex_to_cluster(self, lex, nbits=100):
    #if self.cache.has_key((lex, nbits)):
    #  return self.cache[lex,nbits]
    assert lex[0] == '"' and lex[-1] == '"'
    new_lex = lex[1:-1]
    bitstring = self.clusters.lookup(new_lex)
    res = '"' +"".join(map(str,bitstring[0:nbits])) + '"'
    #self.cache[(lex, nbits)] = res  
    return res
  
class TransFeatureClass(FeatureClass):
  @classmethod
  def make_default(cls):
    return cls()

  @staticmethod
  def join_words(words):
    return "+".join(words)

  @staticmethod
  def crunch(object):
    return str(object).replace(" ", "+")

  def default_counts(self, features):
    return single_count(features)
  
  #TESTING
  test_context = None
  def test_local_feature(self, context):
    return (self.default_counts(self.test_output(context)), self.default_counts(self.extract_local(context)))

  def test_output(self, context):
    raise NotImplementedError, "Look, you gots to implement a test for " + str(self)


class NodeFeatureClass(TransFeatureClass):
  feature_side = LocalContext.SOURCE
  def extract_inner(self, context):
    assert isinstance(context, LocalNodeContext)
    features = self.extract_local(context)
    #assert features, str(self)
    return self.default_counts(features)
  

class LocalFeatureClass(TransFeatureClass):
  def extract_inner(self, context):
    assert isinstance(context, LocalContext)
    features = self.extract_local(context)
    #assert features, str(self)
    return self.default_counts(features)



class GHKMFeatureClass(LocalFeatureClass):
  """
  Feature for each GHKM Rule
  """
  feature_side = LocalContext.SOURCE & LocalContext.TARGET
  prefix = "FullRule"
  def extract_local(self, context):
    return [self.crunch(context.rule.lhs) + "/" + self.join_words(context.cluster_rhs)]

  def test_output(self, context):
    if context.clustering:
      return ['VP(VV+VP(VV+NP+NP(\"¬ÅÊîÅøÅÂ¬Å∫¬ú\")))/x1+\"0010\"+x0+\"0101\"+x2']
    else:
      return ['VP(VV+VP(VV+NP+NP(\"¬ÅÊîÅøÅÂ¬Å∫¬ú\")))/x1+\"that\"+x0+\"considered\"+x2']

    

class BasicFeatureClass(LocalFeatureClass):
  """
  Feature for RHS length
  """
  prefix = "Basic"
  feature_side = LocalContext.TARGET

  def extract_local(self, context):
    return [ f for f in context.fields.items() if ":" not in f[0]]

  def default_counts(self, ls):
    return ls 

  def test_output(self, context): return [('gt_prob',-15.000835),
                                          ('plhs',-6.04973345523),
                                          ('text-length',2),
                                          ('rule-num',1),]


class LexTransFeatures(LocalFeatureClass):
  """
  Feature for lex align
  """
  prefix = "Lex"
  feature_side = LocalContext.TARGET & LocalContext.SOURCE
  
  def extract_local(self, context):
    #print context.fields
    lex_align = [f[0] for f in context.fields.items() if ':' in f[0]]
    #print lex_align
    rhs= [f for f in context.get_fixed_rhs() if is_lex(f)]
    lhs = [f for f in context.get_fringe_lhs() if is_lex(f)]
    ret = []
    for align in lex_align:
      chinese, eng = map(int, align.split(":"))
      ret.append(lhs[chinese] + "+" + rhs[eng])
    return ret
      
      
  def test_output(self, context):
    if context.clustering:
      return [('\"¬ÅÊîÅøÅÂ¬Å∫¬ú\"+"0101"')]
    else:
      return [('\"¬ÅÊîÅøÅÂ¬Å∫¬ú\"+"considered"')]


class SourceCFGFeatureClass(LocalFeatureClass):
  """
  Feature for source CFG Rules
  """
  prefix = "SourceCFG"
  feature_side = LocalContext.SOURCE
  def extract_local(self, context):
    rules = []
    stack = [context.treelet]
    while stack:
      node = stack.pop()
      rules.extend([node.label +"->"+ self.join_words([child.label for child in node.subs])])
      stack.extend([child for child in node.subs if child.subs])
    return  rules

  def test_output(self, context): return ["VP->VV+VP", "VP->VV+NP+NP", "NP->\"¬ÅÊîÅøÅÂ¬Å∫¬ú\""]


class LHSSizeFeatureClass(LocalFeatureClass):
  """ Number of nonterminals in lhs """
  feature_side = LocalContext.SOURCE
  binning = [(0,0), (1,1), (2,2), (3,4), (5,7), (8,100)]
  prefix = "NodeSize"
  def extract_local(self, context):  
    return [self._bin(self._count(context.treelet))]

  @staticmethod
  def _count(tree):
    if is_lex(tree.label):
      return 0
    else :
      return 1 + sum([LHSSizeFeatureClass._count(node) for node in tree.subs])

  def test_output(self, context): return ["5-7"]

class LHSFeatureClass(LocalFeatureClass):
  """Size of the span this rule covers"""
  feature_side = LocalContext.SOURCE
  prefix = "LHS"
  def extract_local(self, context):
    return [self.crunch(context.rule.lhs)]

  def test_output(self, context): return ["VP(VV+VP(VV+NP+NP(\"¬ÅÊîÅøÅÂ¬Å∫¬ú\")))"]

class LHSFringeFeatureClass(LocalFeatureClass):
  """Size of the span this rule covers"""
  feature_side = LocalContext.SOURCE
  prefix = "LHSFringe"
  def extract_local(self, context):
    return [self.join_words(context.get_fringe_lhs())]

  def test_output(self, context): return ["VV+VV+NP+\"¬ÅÊîÅøÅÂ¬Å∫¬ú\""]


# class SizeDiffFeatureClass(LocalFeatureClass):
#   feature_side = LocalContext.SOURCE & LocalContext.TARGET
#   prefix = "SizeDiff"
#   def extract_local(self, context):
#     rhs_words = len([word for word in context.get_fixed_rhs() if is_lex(word)])
#     lhs_words = len([word for word in context.get_fringe_lhs() if is_lex(word)])
#     return [str(rhs_words - lhs_words)]
#   def test_output(self, context):
#     return ["2"]


class RHSFeatureClass(LocalFeatureClass):
  feature_side = LocalContext.TARGET
  prefix = "RHS"
  def extract_local(self, context):
    return [self.join_words(context.get_fixed_rhs())]

  def test_output(self, context):
    if context.clustering:
      return ["VV+\"0010\"+VV+\"0101\"+NP"]
    else:
      return ["VV+\"that\"+VV+\"considered\"+NP"]

# class RHSUnigramClass(LocalFeatureClass):
#   feature_side = LocalContext.TARGET
#   prefix = "UNI"
#   def extract_local(self, context):
#     words = [word for word in context.get_fixed_rhs() if is_lex(word)] 
#     if not words:
#       return ["*BLANK*"]
#     else:
#       return words
#   def test_output(self, context):
#     if context.clustering:
#       return ["\"0010\"", "\"0101\""]
#     else:
#       return ["\"that\"","\"considered\""]

      
class BothFringeFeatureClass(LocalFeatureClass):
  feature_side = LocalContext.TARGET & LocalContext.SOURCE
  prefix = "BothFringe"
  def extract_local(self, context):
    return [self.join_words(context.get_fringe_lhs()) + "/" + self.join_words(context.get_fixed_rhs())]

  def test_output(self, context):
    if context.clustering:
      return ['VV+VV+NP+"\xc2\x81\xe6\x94\x81\xbf\x81\xe5\xc2\x81\xba\xc2\x9c"/VV+"0010"+VV+"0101"+NP']
    else:
      return ['VV+VV+NP+"\xc2\x81\xe6\x94\x81\xbf\x81\xe5\xc2\x81\xba\xc2\x9c"/VV+"that"+VV+"considered"+NP']


class SpanSizeFeatureClass(NodeFeatureClass):
  """Size of the span this rule covers"""
  binning = [(0,0), (1,1), (2,2), (3,5), (6,8), (9,100)]
  prefix = "SpanSize"
  def extract_local(self, context):
    span = context.node.span
    return [self._bin(span[1] - span[0])]

  def test_output(self, context): return ["%s-%s" % (6,8)]


class WordEdgesFeatureClass(NodeFeatureClass):
  feature_side = LocalContext.SOURCE
  prefix = "WordEdge"
  def __init__(self, leftprec=0, leftsucc=0, rightprec=0, rightsucc=0):
    self.leftprec = leftprec
    self.leftsucc = leftsucc
    self.rightprec = rightprec
    self.rightsucc = rightsucc


  @classmethod
  def make_default(cls):
    return cls(1, 1, 1, 1)
        
  def extract_local(self, context):

    seq = []
    
    span = context.node.span
    sent = context.sent + ["_", "_"] ## caution, TODO: should be generic
        
    for i in xrange(1, self.leftprec + 1):
      seq.append(sent[span[0] - i])
      
    for i in xrange(1, self.leftsucc + 1):
      seq.append(sent[span[0] - 1 + i])

    for i in xrange(1, self.rightprec + 1):
      seq.append(sent[span[1] - i])

    for i in xrange(1, self.rightsucc + 1):
      seq.append(sent[span[1] - 1 + i])

    around = "%s/%s/%s/%s" % (self.leftprec, self.leftsucc, self.rightprec, self.rightsucc) 
    return [around + "/W/"+  self.join_words(seq)]    
               
  def test_output(self,context): return ["1/1/1/1/W/¬Å‰Åª¬é¬ÅÊùÅ•+¬Å‰Å∏¬ç¬Å‰Åº¬ö+¬ÅËÅ°¬å¬Å‰Å∏¬Å∫+."]

def get_all_feature_classes(pred_cls):
  "For testing, collect all the feature classes"
  allattrs = [getattr(local_features, fc) for fc in dir(local_features)]
  other = [fc.make_default()
           for fc in allattrs
           if inspect.isclass(fc)
           if issubclass(fc, pred_cls)
           if fc <> pred_cls ]
  return other

class TestLocalFeatureClasses(unittest.TestCase):

    
  def test_all_node_classes(self):
    self.all_classes = get_all_feature_classes(local_features.NodeFeatureClass)
    self.test_context = LocalNodeContext.test_produce_context()
    for feature_classes in self.all_classes:
      (expect, got) = feature_classes.test_local_feature(self.test_context)
      self.assertEqual(got, expect, "Failure in %s. Got: %s, Expect: %s" % (feature_classes.__class__, got, expect))


    
  def test_all_classes(self):
    self.all_classes = get_all_feature_classes(local_features.LocalFeatureClass)
    self.test_context = LocalContext.test_produce_context()
    for feature_classes in self.all_classes:
      (expect, got) = feature_classes.test_local_feature(self.test_context)
      self.assertEqual(got, expect, "Failure in %s. Got: %s, Expect: %s" % (feature_classes.__class__, got, expect))

  def test_cluster_classes(self):
    self.all_classes = get_all_feature_classes(local_features.LocalFeatureClass)
    self.test_context = LocalContext.test_produce_context()

    tdm = TargetDataManager()
    self.test_context.set_cluster_level(tdm, 4)
    for feature_classes in self.all_classes:
      (expect, got) = feature_classes.test_local_feature(self.test_context)
      self.assertEqual(got, expect, "Failure in %s. Got: %s, Expect: %s" % (feature_classes.__class__, got, expect))
      
def test_load(parse_file, trans_file):
  parse_forest = forest.Forest.load(parse_file, False)
  trans_forest = forest.Forest.load(trans_file, True)
 
  #for parse in parse_forest:
  for parse, trans in izip(parse_forest, trans_forest):
    fe = FeatureExtractor(get_all_feature_classes())

    for node in trans.nodes.values():      
      for edge in node.edges:
        local_context = LocalContext(node, edge, edge.rule, trans.sent)
        features = fe.extract_all(local_context)
        # print features

        edge.fvector = " ".join([ f  for f in features])  #dict([ (f,1)  for f in features]) 
        edge.rule.fields = Vector(" ".join(features))
        
    trans.dump()

    #for _, rule in trans.edges.items():
     # print rule.lhs, rule.rhs, rule.fields


if __name__=="__main__": unittest.main()
