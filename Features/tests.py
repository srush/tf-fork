import sys
sys.path.append("..")
from forest import Forest
from StringIO import StringIO
f = open("/tmp/features_oracle")
for l in f:
  sent, _ = l.split("****")
  s2 = sent.replace("\t\t\t", "\n")
  f = open("/tmp/blah4", 'w')
  f.write(s2)
  f.close()
  for sent_forest in Forest.load(StringIO(s2), True, lm=None):
    sent_forest.dump()











