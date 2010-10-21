#!/usr/bin/env python

'''a wrapper/reimplementation of hiero monitor. added global information like free memory.
'''

#import mycode
import sys, os, gc
import monitor

logs = sys.stderr

def cpu():
    return monitor.cpu()


proc_status = '/proc/%d/status' % os.getpid()
mem_info = '/proc/meminfo'

scale = {'B': 1.0,
         'kB': 1024.0, 'mB': 1024.0*1024.0,
         'KB': 1024.0, 'MB': 1024.0*1024.0,
         'gB': 1024*1024*1024.0, 'GB': 1024*1024*1024.0}

def human(v):
    for a in ['GB', 'MB', 'kB']:
        if v > scale[a]:
            return "%3.1lf %s" % (v/scale[a], a)
    return "%3.1lf B" % v
    
class Mem(object):
    __slots__ = "v", "s"
    def __init__(self, v=0, s=""):
        self.v = v
        if s == "":
            s = human(v)
        self.s = s

    def __str__(self):
        try:
            return "%.0f (%s)" % (self.v, self.s)
        except:
            print >> logs, "ERROR: v=", self.v, "s=", self.s

    def __sub__(self, other):
        assert isinstance(other, Mem)
        return Mem(self.v - other. v)

    def __cmp__(self, other):
        if type(other) in [float, int]:
            return self.v - other
        else:
            assert isinstance(other, Mem)
            return self.v - other.v

MemZero = Mem(0.0)

def VmB(secret_file, VmKey):
    '''Private.
    '''
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(secret_file)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
    return Mem(float(v[1]) * scale[v[2]])

def memory(since=MemZero):
    ''' return a class of float and str (pp). Note: monitor.memory returns only a float'''
    return VmB(proc_status, "VmSize:") - since

def freemem():
    return VmB(mem_info, "MemFree:")
    
def gc_collect():
    before = memory()
    gc.collect()
    after = memory()
    print >> logs, "(%s collected in gc)" % (before - after)
    
