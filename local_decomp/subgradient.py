from svector import Vector
import time
class SubgradientSolver(object):
  def __init__(self, s1): 
    self.weights = Vector()
    self.lowest_primal = 1e20
    self.highest_dual = 0.0
    self.round = 1
    self.nround = 1
    self.s1 = s1


  def run(self):
    self.duals = []
    while self.run_one_round():
      pass


  def run_one_round(self):
    start = time.time()

    dstart = time.time()
    
    (subgrad, dual, primal) = self.s1.decode()
    self.duals.append(dual)
    dend = time.time()
    print "decode time", dend -dstart
    print "\n\n\n"

       
    print "Round : %s"% self.round
    #print "Dual  %s %s %s" % (obj, obj2, str(obj2+obj))

    
    
    change = sum([abs(subgrad[f]) for f in subgrad])    

    print "CHANGE IS %s"%change

    self.update_weights(subgrad)
    end = time.time()
    print "One Round", end -start
    return (change <> 0.0 and self.round < 1000)

   
  def update_weights(self, subgrad):
    self.base_weight = 1
      
    self.round += 1
    if len(self.duals) > 2 and self.duals[-1] < self.duals[-2]:
      self.nround += 1 


    alpha = (self.base_weight) * (0.98) ** (5* float(self.nround))
      
    updates = alpha * subgrad
        
    self.weights += updates

    self.s1.delta_weights(updates, self.weights)

