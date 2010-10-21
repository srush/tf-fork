#!/usr/bin/env python

import sys
import itertools
from svector import Vector
'''"!"" -> "!" "\"" "by" "xiong" "zhongmao" ### gt_prob=-10 proot=-10 prhs=0.0000 plhs=-4.5850 lexpef=-34.9396 lexpfe=-0.8863
 '''
def input():
    ws = Vector('gt_prob=-1.0 proot=-1.0 prhs=-1.0 plhs=-1.0 lexpef=-1.0 lexpfe=-1.0')
    for line in sys.stdin:
        chi, other = line.split(' -> ', 1)
        eng, fields = other.split(' ### ', 1)
        features = Vector(fields)
        score = features.dot(ws)
        yield (score, chi, line)


if __name__ == "__main__":
    k = int(sys.argv[1])

    for key, val in itertools.groupby(input(), lambda x: x[1]):
        for (mscore, _, mline) in sorted( (score, chi, line) for (score, chi, line) in val)[0:k]:
             #print "%4.lf\t%s" % (mscore, mline.strip())
             print mline.strip()
