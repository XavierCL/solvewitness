from AbstractProblemDefinition import AbstractProblemDefinition
import numpy as np

from utils import s2c, shift, arrayToDebug

class ProblemDefinition(AbstractProblemDefinition):
  def __init__(self, starting: np.ndarray):
    self.starting = starting

  def getStarting(self):
    startingMask = self.starting == s2c('s')
    fakeStarter = np.zeros_like(self.starting)
    fakeStarter[startingMask] = s2c('o')
    startingPositions = []
    for startingIndex in np.argwhere(startingMask):
      sampleStarting = np.copy(fakeStarter)
      sampleStarting[tuple(startingIndex)] = s2c('h')
      startingPositions.append(sampleStarting)
      
    return startingPositions

  def getNexts(self, current):
    hMask = current == s2c('h')
    oMask = current == s2c('o')
    hToP = np.where(np.any([hMask, oMask], 0), s2c('p'), current)
    nextStates = []

    nextMultiHMasks = [
      np.all([shift(hMask, (1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e'), self.starting == s2c('b')], 0)], 0),
      np.all([shift(hMask, (1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e'), self.starting == s2c('b')], 0)], 0),
      np.all([shift(hMask, (-1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e'), self.starting == s2c('b')], 0)], 0),
      np.all([shift(hMask, (-1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e'), self.starting == s2c('b')], 0)], 0)
    ]

    nextMultiOMasks = [
      np.all([shift(oMask, (-1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e'), self.starting == s2c('y')], 0)], 0),
      np.all([shift(oMask, (-1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e'), self.starting == s2c('y')], 0)], 0),
      np.all([shift(oMask, (1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e'), self.starting == s2c('y')], 0)], 0),
      np.all([shift(oMask, (1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e'), self.starting == s2c('y')], 0)], 0)
    ]

    for nextMultiHMask, nextMultiOMask in zip(nextMultiHMasks, nextMultiOMasks):
      if not np.any(nextMultiHMask) or not np.any(nextMultiOMask):
        continue

      if np.all([nextMultiHMask == nextMultiOMask]):
        continue

      nextState = np.copy(hToP)
      nextState[nextMultiHMask] = s2c('h')
      nextState[nextMultiOMask] = s2c('o')
      nextStates.append(nextState)

    return nextStates

  def isSatisfied(self, current):
    allHeads = np.any([current == s2c('h'), current == s2c('o')], 0)
    allEnds = self.starting == s2c('e')
    allHeadsAreAtEnd = np.all([allHeads == np.all([allHeads, allEnds], 0)])

    allColors = np.any([self.starting == s2c('b'), self.starting == s2c('y')], 0)
    allPathOrHeads = np.any([allHeads, current == s2c('p')], 0)
    allColorsAreCovered = np.all([allColors == np.all([allColors, allPathOrHeads], 0)])
    return allHeadsAreAtEnd and allColorsAreCovered
  
  def isUnsatisfiable(self, current):
    return False