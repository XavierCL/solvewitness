from AbstractProblemDefinition import AbstractProblemDefinition
import numpy as np

from utils import s2c, shift, arrayToDebug

class ProblemDefinition(AbstractProblemDefinition):
  def __init__(self, starting: np.ndarray):
    self.starting = starting
    middleSquares = starting[slice(1,-1,2),slice(1,-1,2)]
    self.spikes = np.array([middleSquares == l for l in globalSpikes])
    self.squareColors = np.array([middleSquares == l for l in globalSquares])
    self.esMask = starting == s2c('e')
    self.es = np.argwhere(self.esMask)
    self.geos = {s2c(geo[0]): self.buildGeo(geo) for geo in globalGeoStore}
    self.onlyGeos = middleSquares
    self.onlyGeos[~np.isin(middleSquares, list(self.geos.keys()))] = 0

    self.spikeColorPairMasks = np.copy(self.squareColors)

    for geoDefinition in globalGeoStore:
      if geoDefinition[1] == 0:
        continue

      spikeIndex = np.argwhere(globalSpikes == s2c(geoDefinition[1]))[0][0]
      self.spikeColorPairMasks[spikeIndex] = np.any([self.spikeColorPairMasks[spikeIndex], self.onlyGeos == s2c(geoDefinition[0])], 0)

  # To support mishaps, just convert the mishap to a new spike with all other items
  # To make the computation faster, prebuild all possible geo arangements, called mandatory lines
  # Also add neighbouring color squares to those mandatory lines
  # Also create negative mandatory lines, where no line can ever pass
  # Also create mandatory points on points
  # For each possible set of mandatory lines, make sure they are not unsatisfiable
  # Unsatisfiability includes impossible path crossings, X shape future paths are impossible
    # How to detect Xs? Only the next bordering mandatory line or a middle mandatory line are allowed
  # Run the brute force for each combination of paths.
  # Add to the unsatisfiability that each pair of line must be reachable after every move.
  # Sort the next states by distance to the next mandatory line instead of the end
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

    realStarting.sort(key=lambda x: self.evalRemaining(x))
      
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
    headZoneIndices = self.fullMapToZoneIndex(np.argwhere(current == s2c('h')), zoneIndices)
    endZoneIndices = self.fullMapToZoneIndex(self.es, zoneIndices)

    # If the head can't reach any end
    if np.intersect1d(endZoneIndices, headZoneIndices).size == 0:
      return True
    
    # If the zones out of head reach are unsatisfied
    for satisfiableZoneIndex in range(np.max(zoneIndices) + 1):
      if np.any(headZoneIndices == satisfiableZoneIndex):
        if self.isPendingZoneUnsatisfiable(zoneIndices, satisfiableZoneIndex):
          return True

      elif not self.isZoneSatisfied(zoneIndices, satisfiableZoneIndex):
        return True

    return False
  
  def evalRemaining(self, current):
    hPos = np.argwhere(current == s2c('h'))[0]
    return int(np.min(np.sum(np.abs(self.es - hPos), axis=1)))
  
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

    # Square handling
    squareColorsMask = np.all([self.squareColors, np.repeat(zoneMask[np.newaxis,:,:], self.squareColors.shape[0], axis=0)], 0)

    if np.count_nonzero(np.any(squareColorsMask, axis=(1, 2))) > 1:
      return False

    # Spike handling
    spikesMask = np.all([self.spikes, np.repeat(zoneMask[np.newaxis,:,:], self.spikes.shape[0], axis=0)], 0)
    spikesCount = np.count_nonzero(spikesMask, axis=(1, 2))

    spikeColorPairMask = np.all([self.spikeColorPairMasks, np.repeat(zoneMask[np.newaxis,:,:], self.spikeColorPairMasks.shape[0], axis=0)], 0)
    spikeColorPairCount = np.count_nonzero(spikeColorPairMask, axis=(1, 2))

    totalSpikeCounts = spikesCount + spikeColorPairCount

    if np.any(np.all([spikesCount > 0, totalSpikeCounts != 2], 0)):
      return False
    
    # Geo
    zonedGeosInMap = np.copy(self.onlyGeos)
    zonedGeosInMap[~zoneMask] = 0
    zonedGeos = zonedGeosInMap[zonedGeosInMap != 0]
    zonedGeos = [self.geos[geoChar] for geoChar in zonedGeos]

    if len(zonedGeos) > 0:

      nonRotatedZonedGeos = [z[0] for z in zonedGeos]

      if np.count_nonzero(zoneMask) != np.count_nonzero(nonRotatedZonedGeos):
        return False
      
      if not self.recursiveCanPlaceAllGeos(zoneMask, zonedGeos):
        return False
    
    return True
  
  def isPendingZoneUnsatisfiable(self, zoneIndices, zoneIndex):
    zoneMask = (zoneIndices == zoneIndex)

    # Squares are always eventually satisfiable
    # Spikes need to be in pairs
    spikesMask = np.all([self.spikes, np.repeat(zoneMask[np.newaxis,:,:], self.spikes.shape[0], axis=0)], 0)
    spikesCount = np.count_nonzero(spikesMask, axis=(1, 2))

    spikeColorPairMask = np.all([self.spikeColorPairMasks, np.repeat(zoneMask[np.newaxis,:,:], self.spikeColorPairMasks.shape[0], axis=0)], 0)
    spikeColorPairCount = np.count_nonzero(spikeColorPairMask, axis=(1, 2))

    totalSpikeCounts = spikesCount + spikeColorPairCount

    if np.any(np.all([spikesCount > 0, totalSpikeCounts % 2 == 1], 0)):
      return True
    
    # Geos need to have a minimum of space
    zonedGeosInMap = np.copy(self.onlyGeos)
    zonedGeosInMap[~zoneMask] = 0
    zonedGeos = zonedGeosInMap[zonedGeosInMap != 0]
    zonedGeos = [self.geos[geoChar] for geoChar in zonedGeos]

    if len(zonedGeos) > 0:

      nonRotatedZonedGeos = [z[0] for z in zonedGeos]

      if np.count_nonzero(zoneMask) < np.count_nonzero(nonRotatedZonedGeos):
        return True
      
    return False
  
  def recursiveCanPlaceAllGeos(self, zoneMaskLeft, geosLeftToPlace):
    if len(geosLeftToPlace) == 1:
      return len(self.getConfigurations(zoneMaskLeft, geosLeftToPlace[0], earlyReturn=True)) > 0
    
    geoConfigurations = []

    for geo in geosLeftToPlace:
      configurations = self.getConfigurations(zoneMaskLeft, geo, earlyReturn=False)
      if len(configurations) == 0:
        return False
      
      geoConfigurations.append((configurations, geo))

    geoConfigurations.sort(key=lambda x: len(x[0]))

    newRemainingGeos = [x[1] for x in geoConfigurations[1:]]
    
    for geoConfiguration in geoConfigurations[0][0]:
      newZoneMask = np.all([zoneMaskLeft, ~geoConfiguration], 0)
      if self.recursiveCanPlaceAllGeos(newZoneMask, newRemainingGeos):
        return True
      
    return False
  
  def getConfigurations(self, zoneMask, geo, earlyReturn):
    whereZoneMask = np.argwhere(zoneMask)
    topZoneMask = np.min(whereZoneMask[:,0])
    bottomZoneMask = np.max(whereZoneMask[:,0]) + 1
    leftZoneMask = np.min(whereZoneMask[:,1])
    rightZoneMask = np.max(whereZoneMask[:,1]) + 1
    zoneMaskHeight = bottomZoneMask - topZoneMask
    zoneMaskWidth = rightZoneMask - leftZoneMask

    configurations = []

    for angledGeo in geo:
      whereGeo = np.argwhere(angledGeo)
      topGeo = np.min(whereGeo[:,0])
      bottomGeo = np.max(whereGeo[:,0]) + 1
      leftGeo = np.min(whereGeo[:,1])
      rightGeo = np.max(whereGeo[:,1]) + 1
      geoHeight = bottomGeo - topGeo
      geoWidth = rightGeo - leftGeo

      if zoneMaskHeight < geoHeight or zoneMaskWidth < geoWidth:
        continue
      
      for topY in range(topZoneMask, bottomZoneMask - geoHeight + 1):
        for leftX in range(leftZoneMask, rightZoneMask - geoWidth + 1):
          shiftedAngledGeo = shift(angledGeo, (topY, leftX), (0, 1), 0)
          if np.all(np.all([shiftedAngledGeo, zoneMask], 0) == shiftedAngledGeo):
            configurations.append(shiftedAngledGeo)

            if earlyReturn:
              return np.array(configurations, dtype=np.bool)

    return np.array(configurations, dtype=np.bool)

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
  
  def fullMapToZoneIndex(self, indices, zoneIndices):
    multiIndexOnFullMap = np.repeat(indices[:,np.newaxis,:], 4, axis=1)
    multiIndexOnFullMap += [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    multiIndexOnFullMap = multiIndexOnFullMap.reshape((-1, 2))
    multiIndexOnFullMapCorrectMask = np.all([multiIndexOnFullMap < self.starting.shape, multiIndexOnFullMap >= 0], (0, 2))
    multiIndexOnFullMap = multiIndexOnFullMap[multiIndexOnFullMapCorrectMask]
    multiHeadIndexOnZoneMap = (multiIndexOnFullMap / 2).astype(np.int64)
    return zoneIndices[tuple(multiHeadIndexOnZoneMap.T)]
  
  # Discard duplicated rotations
  def buildGeo(self, geoDefinition):
    (_char, _pairedSpike, rotate, array) = geoDefinition

    array = np.array(array)
    smallArrays = [array]

    if rotate:
      smallArrays = [
        array,
        np.flip(array, (0, 1)),
        np.flip(array.T, (0,)),
        np.flip(array.T, (1,)),
      ]

    squareShape = ((np.array(self.starting.shape) - 1) / 2).astype(np.int32)
    smallArrays = [s for s in smallArrays if s.shape[0] <= squareShape[0] and s.shape[1] <= squareShape[1]]

    bigArrays = [np.concatenate([np.concatenate([x, np.zeros((squareShape[0] - x.shape[0], x.shape[1]))], 0), np.zeros((squareShape[0], squareShape[1] - x.shape[1]))], 1) for x in smallArrays]

    return bigArrays
  
# map letter, linked spike, rotatable, matrix definition
globalGeoStore = [
  (
    'a',
    'z',
    False,
    [
      [True],
      [True],
      [True],
    ]
  ),(
    'b',
    0,
    False,
    [
      [True, True, True],
    ]
  ),(
    'c',
    0,
    False,
    [
      [True, True, True],
      [False, False, True],
    ]
  ),(
    'd',
    0,
    False,
    [
      [True],
    ]
  ),(
    'f',
    0,
    False,
    [
      [True, True],
    ]
  )
]

globalSpikes = np.array([s2c(l) for l in ['z', 'y', 'x']])
globalSquares = np.array([s2c(l) for l in ['l', 'm', 'n']])