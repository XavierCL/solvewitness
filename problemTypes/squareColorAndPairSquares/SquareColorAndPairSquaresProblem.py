from AbstractProblemDefinition import AbstractProblemDefinition
import numpy as np

from utils import s2c, shift, arrayToDebug

class ProblemDefinition(AbstractProblemDefinition):
  def __init__(self, starting: np.ndarray):
    self.starting = starting
    self.spikes = np.array([starting[slice(1,-1,2),slice(1,-1,2)] == s2c(l) for l in ['z']])
    self.squareColors = np.array([starting[slice(1,-1,2),slice(1,-1,2)] == s2c(l) for l in ['o', 'v']])
    self.es = np.argwhere(starting == s2c('e'))
    self.esMask = starting == s2c('e')

  def getStarting(self):
    startingMask = self.starting == s2c('s')
    fakeStarter = np.zeros_like(self.starting)
    startingPositions = []
    for startingIndex in np.argwhere(startingMask):
      sampleStarting = np.copy(fakeStarter)
      sampleStarting[tuple(startingIndex)] = s2c('h')
      startingPositions.append(sampleStarting)

    realStarting = []
    for startingPosition in startingPositions:
      smallNexts = self.getSmallNexts(startingPosition)
      if len(smallNexts) > 0:
        realStarting += smallNexts
      else:
        realStarting.append(startingPosition)
      
    return realStarting

  def getNexts(self, current):
    hMask = current == s2c('h')
    hToP = np.where(hMask, s2c('p'), current)
    nextStates = []

    nextMultiHMasks = [
      np.all([shift(hMask, (1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e')], 0)], 0)
    ]

    next2MultiHMasks = [
      np.all([shift(hMask, (2,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (2,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-2,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('|'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-2,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('-'), self.starting == s2c('e')], 0)], 0)
    ]

    for nextMultiHMask, next2MultiHMask in zip(nextMultiHMasks, next2MultiHMasks):
      if not np.all(np.any([nextMultiHMask, next2MultiHMask], axis=(1, 2))):
        continue

      nextState = np.copy(hToP)
      nextState[nextMultiHMask] = s2c('p')
      nextState[next2MultiHMask] = s2c('h')
      nextStates.append(nextState)

    nextStates.sort(key=lambda x: self.evalRemaining(x))

    return nextStates

  def isSatisfied(self, current):
    if not np.any(np.all([np.any([current == s2c('h'), current == s2c('p')], 0), self.esMask], 0)):
      return False

    zoneIndices = self.getZoneIndices(current)

    for satisfiableZoneIndex in range(np.max(zoneIndices) + 1):
      if not self.isZoneSatisfied(zoneIndices, satisfiableZoneIndex):
        return False
      
    return True
  
  def isUnsatisfiable(self, current):
    # If all ends are a path
    pathMask = current == s2c('p')
    if np.all(pathMask[self.esMask]):
      return True
    
    # If the head can't go nowhere
    headMask = current == s2c('h')
    walkablePaths = np.all([~pathMask, np.any([self.starting == s2c('|'), self.starting == s2c('-'), self.starting == s2c('+'), self.starting == s2c('e')], 0)], 0)
    nextSteps = np.any([shift(headMask, (1,), (0,), 0),shift(headMask, (-1,), (0,), 0), shift(headMask, (1,), (1,), 0), shift(headMask, (-1,), (1,), 0)], 0)
    if not np.any(np.all([nextSteps, walkablePaths], 0)):
      return True

    zoneIndices = self.getZoneIndices(current)
    headIndexOnFullMap = np.argwhere(current == s2c('h'))[0]
    headIndexOnZoneMap = ((headIndexOnFullMap - 1) / 2).astype(np.int64)
    endIndicesOnZoneMap = ((self.es - 1) / 2).astype(np.int64)

    headZoneIndex = zoneIndices[tuple(headIndexOnZoneMap)]
    endZoneIndices = zoneIndices[tuple(endIndicesOnZoneMap.T)]

    # If the head can't reach any end
    if np.all(endZoneIndices != headZoneIndex):
      return True
    
    # If the zones out of head reach are unsatisfied
    for satisfiableZoneIndex in range(np.max(zoneIndices) + 1):
      if satisfiableZoneIndex == headZoneIndex:
        continue

      if not self.isZoneSatisfied(zoneIndices, satisfiableZoneIndex):
        return True

    return False
  
  def evalRemaining(self, current):
    hPos = np.argwhere(current == s2c('h'))[0]
    return np.min(np.sum(np.abs(self.es - hPos), axis=1))
  
  def getZoneIndices(self, current):
    leftBlockers = np.any([current == s2c('p'), current == s2c('h')], 0)[slice(1, None, 2), slice(0, -2, 2)]
    rightBlockers = np.any([current == s2c('p'), current == s2c('h')], 0)[slice(1, None, 2), slice(2, None, 2)]
    topBlockers = np.any([current == s2c('p'), current == s2c('h')], 0)[slice(2, None, 2), slice(1, None, 2)]
    bottomBlockers = np.any([current == s2c('p'), current == s2c('h')], 0)[slice(0, -2, 2), slice(1, None, 2)]

    zoneIndices = np.ones((int(current.shape[0] / 2), int(current.shape[1] / 2)), dtype=np.int32) * -1
    lastZoneIndex = 0

    while np.any(zoneIndices == -1):
      unassignedIndices = np.argwhere(zoneIndices == -1)
      victimSquareIndex = unassignedIndices[int(unassignedIndices.shape[0]/1.7)]

      previousZoneIndices = np.copy(zoneIndices)
      zoneIndices[tuple(victimSquareIndex)] = lastZoneIndex

      while np.any(zoneIndices != previousZoneIndices):
        previousZoneIndices = np.copy(zoneIndices)
        zoneIndices[np.all([zoneIndices == -1, shift(zoneIndices, (1,), (1,), -1) == lastZoneIndex, ~leftBlockers], 0)] = lastZoneIndex
        zoneIndices[np.all([zoneIndices == -1, shift(zoneIndices, (-1,), (1,), -1) == lastZoneIndex, ~rightBlockers], 0)] = lastZoneIndex
        zoneIndices[np.all([zoneIndices == -1, shift(zoneIndices, (1,), (0,), -1) == lastZoneIndex, ~bottomBlockers], 0)] = lastZoneIndex
        zoneIndices[np.all([zoneIndices == -1, shift(zoneIndices, (-1,), (0,), -1) == lastZoneIndex, ~topBlockers], 0)] = lastZoneIndex
      
      lastZoneIndex += 1

    return zoneIndices
    
  def isZoneSatisfied(self, zoneIndices, zoneIndex):
    zoneMask = (zoneIndices == zoneIndex)
    spikesMask = np.all([self.spikes, np.repeat(zoneMask[np.newaxis,:,:], self.spikes.shape[0], axis=0)], 0)
    spikesCount = np.count_nonzero(spikesMask, axis=(1, 2))

    if np.any(np.all([spikesCount != 0, spikesCount != 2], 0)):
      return False
    
    squareColorsMask = np.all([self.squareColors, np.repeat(zoneMask[np.newaxis,:,:], self.squareColors.shape[0], axis=0)], 0)

    if np.count_nonzero(np.any(squareColorsMask, axis=(1, 2))) > 1:
      return False
    
    return True

  def getSmallNexts(self, current):
    hMask = current == s2c('h')
    hToP = np.where(hMask, s2c('p'), current)
    nextStates = []

    nextMultiHMasks = [
      np.all([shift(hMask, (1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-1,), (0,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('e')], 0)], 0),
      np.all([shift(hMask, (-1,), (1,), 0), hToP != s2c('p'), np.any([self.starting == s2c('+'), self.starting == s2c('e')], 0)], 0)
    ]

    for nextMultiHMask in nextMultiHMasks:
      if not np.any(nextMultiHMask):
        continue

      nextState = np.copy(hToP)
      nextState[nextMultiHMask] = s2c('h')
      nextStates.append(nextState)

    nextStates.sort(key=lambda x: self.evalRemaining(x))

    return nextStates