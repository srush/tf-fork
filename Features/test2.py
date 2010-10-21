for l in open("tmp/weights.round.0.0.412698074071"):
  if l.strip():
    if len(l.strip().split("\t")) <> 2:
      print >>sys.stderr, l
      assert False
    feat, weight = l.strip().split("\t")
    #weights[feat] = float(weight)
