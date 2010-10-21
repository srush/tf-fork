#!/usr/bin/env python

'''generic averaged perceptron trainer.'''

from __future__ import division

import math
import sys
from trainer import BasePerceptron, Perceptron
logs = sys.stderr

from collections import defaultdict

from svector import Vector

import time


import gflags as flags

FLAGS = flags.FLAGS

class DistributedPerceptron(BasePerceptron):

  def set_training(self, training):
    self.trainfiles = training
    
  def train(self):
    "Just train for one round and dump weights, no eval"
    starttime = time.time()
    print >> logs, "starting perceptron at", time.ctime()
    num_updates, early_updates = self.one_pass_on_train()
    self.decoder.set_model_weights(self.weights) # restore non-avg
    self.dump(self.weights)

    
  @staticmethod
  def cmdline_perc(decoder):
    
    dist =  DistributedPerceptron(decoder, train = [FLAGS.train],
                      output = FLAGS.out)
    #dist.load_parameters(FLAGS.parameters)
    return dist

class MergePerceptron(Perceptron):
  @staticmethod
  def merge_as_files(files):
    weights = Vector()
    for file in files:
      for f in open(file):
        if not f.strip():
          continue
        feat, up = f.strip().split("\t")
        weights[feat] += float(up)
    return weights

  @staticmethod
  def merge_as_reduce(stdin):
    weights = Vector()
    for l in stdin:
      if not l.strip(): continue
      val, key = l.strip().split("\t")
      weights[val] += float(key)
    return weights

  @staticmethod
  def dump_weights(weights):
    for k, v in weights.iteritems():
      print "%s\t%s"%(k,v)
      
def main():
  if sys.argv[1] == "file":
    weights = MergePerceptron.merge_as_files(sys.argv[2:])
    MergePerceptron.dump_weights(weights)
  else:
    weights = MergePerceptron.merge_as_reduce(sys.stdin)
    MergePerceptron.dump_weights(weights)


if __name__=="__main__":
  main()
