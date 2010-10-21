import sys

for l in sys.stdin:
  if not l.strip(): continue
  k, v = l.strip().split("\t")
  print "%s=%s"%(k,v),
