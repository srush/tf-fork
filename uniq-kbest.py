#!/usr/bin/env python

''' remove duplicates in a kbest list, keeping order.'''

import sys

if __name__ == "__main__":

    cache = set()
    for line in sys.stdin:
        line = line.strip()
        if line.find("segid=") >= 0:
            if cache != set():
                print >> sys.stderr, "%s: %d unique." % (lastsent, len(cache))                
            cache = set()
            lastsent = line
        else:
            if line not in cache:
                cache.add(line)
                print line

    print >> sys.stderr, "%s: %d unique." % (lastsent, len(cache))
        
            
    
