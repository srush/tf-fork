
class FeatureExtractor(object):
  """
  Controller for managing feature extraction.
  
  Takes a "context" - opaque data  
  kk"""
  def __init__(self, classes):
    self.feature_classes = classes

  def extract_all(self, context):
    return sum([
      feature_class.extract(context)
      for feature_class in self.feature_classes], [])

  def dump_active_feature_classes():
    "\n".join([fc.prefix for fc in self.feature_classes])

class FeatureClass(object):
  """
  A feature class is in charge of extracting features of a certain type from a context.
  To use: Implement extract_inner and self.prefix
  """
  
  def extract(self, context):
    features = self.extract_inner(context)
    assert self.prefix, "FeatureClasses need a prefix"  
    return [(self.prefix + "/" + feature[0] + "=" +str(feature[1]))
            for feature in features]

  @staticmethod
  def _check_feature(feature):
    assert " " not in feature 
    return feature
  
  def _extract_inner(self, context):
    raise NotImplementedError, "abstract"

  def _bin(self, num):
    assert self.binning
    for (start, end) in self.binning:
      if start <= num <= end:
        return "%s-%s"% (start, end)
    assert AssertionError, "Binning Failed " + str(self.binning)
    
  @staticmethod
  def description():
    raise NotImplementedError, "abstract"

class SampleFeatureClass(FeatureClass):
  """
  Features that match Words given in constructor 
  """
  prefix = "Match"
  
  def __init__(self, matches):
    self.matches = matches
    
    
  def _extract_inner(self, context):
    return [m for m in matches
              if m == context]

