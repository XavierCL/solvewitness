from abc import ABC

class AbstractProblemDefinition(ABC):
  def getStarting(self):
    pass

  def getNexts(self, current):
    pass

  def isSatisfied(self, current):
    return False
  
  def isUnsatisfiable(self, current):
    return True