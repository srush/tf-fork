# -*- coding: utf-8 -*-
"""
Clusters for features
"""

import unittest

class Clusters(object):

  def __init__(self, clusters):
    self.clusters = clusters

  def lookup(self, word):
    if self.clusters.has_key(word):
      return self.clusters[word]
    else:
      return []
  @classmethod
  def read_clusters(cls, file):
    clusters = {}
    for l in open(file):
      bitstring, word, counts= l.strip().split()
      if word.islower() or not clusters.has_key(word.lower()):
        clusters[word.lower()] = map(int, bitstring)
      
    return cls(clusters)

class TestCluster(unittest.TestCase):
  def test_clusters(self):
    clust = Clusters.read_clusters("/home/nlg-02/ar_009/paths")
    self.assertEqual(clust.lookup("uncomic"), [0,0,1,0,1,0,0,1,0,0])


if __name__=="__main__": unittest.main()
