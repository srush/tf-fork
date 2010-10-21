"""
Hadoop code to compute marginals
"""

import sys
import os
import re
import subprocess
import logging
import itertools
from svector import Vector
from StringIO import StringIO
code_dir = os.getenv("TRANSFOREST")
sys.path.append(code_dir)
sys.path.append(code_dir+'/scripts/')
sys.path.append(code_dir+'/Features/')
sys.path.append(code_dir+'/wrappers/')
from shell_helpers import *
from encode import *
from forest import Forest
import fast_inside_outside

class MarginalMap(object):
  def __init__(self):
    #load weights
    self.weights = Vector()
    for l in open(os.getenv("WEIGHTS")):
      if l.strip():
        #if len(l.strip().split("\t")) <> 2:
          #print >>sys.stderr, l
        feat, weight = l.strip().split("\t")
        if abs(float(weight)) > 1e-10:
          self.weights[feat] = float(weight)
    self.processed = self.counters["Processed"]

  def __call__(self, data):
    vec = Vector()
    for i, (key, val) in enumerate(data):
      splits = val.split("****")
      if len(splits) <> 2:
        print >>sys.stderr,"skipping sent"
        continue
      sent, oracle = splits 
      s2 = sent.replace("\t\t\t", "\n")
      o2 = oracle.replace("\t\t\t", "\n")
      sent_forest = Forest.load(StringIO(s2), True, lm=None).next()
      oracle_forest = Forest.load(StringIO(o2), True, lm=None).next()
      assert sent_forest, oracle_forest
      #print >>sys.stderr, len(sent_forest)
      #print >>sys.stderr, len(oracle_forest)
      example_marg, example_partition  = fast_inside_outside.collect_marginals(sent_forest, self.weights)
      oracle_marg, oracle_partition  = fast_inside_outside.collect_marginals(oracle_forest, self.weights)
      vec += example_marg - oracle_marg
      vec["log_likelihood"] += example_partition-oracle_partition
      #vec["log_likelihood"] += example_partition-oracle_partition 
      self.processed += 1
    for feat in vec:
      yield feat, vec[feat]
      
def sum_reduce(key, vals):
  yield key, sum(vals)


def starter(program):
  def require(a):
    var = program.delopt(a)
    assert var, a
    return var
  
  weight_file = require("weights")
    
  #program.addopt("cacheFile",weight_file)
  program.addopt("cmdenv","WEIGHTS=%s"%weight_file)
  program.addopt("cmdenv","PYTHON=%s"%os.getenv("PYTHON"))
  program.addopt("cmdenv","TRANSFOREST=%s"%os.getenv("TRANSFOREST"))
  program.addopt("cmdenv","LD_LIBRARY_PATH=%s"%os.getenv("LD_LIBRARY_PATH"))
  program.addopt("cmdenv","PYTHONPATH=%s"%os.getenv("PYTHONPATH"))  
  

def runner(job):  
  import dumbo
  dumbo.run(MarginalMap, sum_reduce)


if __name__ == "__main__":
  import dumbo
  dumbo.main(runner,starter)
