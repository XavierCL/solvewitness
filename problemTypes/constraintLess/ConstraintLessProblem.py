from AbstractProblemDefinition import AbstractProblemDefinition
import numpy as np

from utils import s2c, shift, arrayToDebug

class ProblemDefinition(AbstractProblemDefinition):
  def __init__(self, starting: np.ndarray):
    self.starting = starting

  def getStarting(self):
    return [np.where(self.starting == s2c('s'), s2c('h'), 0)]

  def getNexts(self, current):
    hMask = current == s2c('h')
    hToP = np.where(hMask, s2c('p'), current)
    nextHMasks = []

    nextMultiHMasks = [
      np.all([shift(hMask, (1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e')], 0)], 0)
    ]

    for nextMultiHMask in nextMultiHMasks:
      for hArg in np.argwhere(nextMultiHMask):
        nextHMask = np.copy(hToP)
        nextHMask[tuple(hArg)] = s2c('h')
        nextHMasks.append(nextHMask)

    return nextHMasks

  def isSatisfied(self, current):
    return np.any([np.all([current == s2c('h'), self.starting == s2c('e')], 0)])
  
  def isUnsatisfiable(self, current):
    return False