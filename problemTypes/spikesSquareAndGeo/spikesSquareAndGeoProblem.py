from AbstractProblemDefinition import AbstractProblemDefinition
import numpy as np

from utils import pad, prepad, s2c, shift, arrayToDebug

class ProblemDefinition(AbstractProblemDefinition):
  def __init__(self, starting: np.ndarray):
    self.starting = starting
    middleSquares = starting[slice(1,-1,2),slice(1,-1,2)]
    self.spikes = np.array([middleSquares == l for l in globalSpikes])
    self.squareColors = np.array([middleSquares == l for l in globalSquares])
    self.esMask = starting == s2c('e')
    self.es = np.argwhere(self.esMask)
    self.geos = {s2c(geo[0]): (geo[3], self.buildGeo(geo)) for geo in globalGeoStore}
    self.onlyGeos = middleSquares
    self.onlyGeos[~np.isin(middleSquares, list(self.geos.keys()))] = 0
    self.points = np.array([starting == l for l in globalPoints])
    self.verticalPaths = np.concatenate([[s2c(l) for l in ['+','|','e']], globalPoints])
    self.horizontalPaths = np.concatenate([[s2c(l) for l in ['+','-','e']], globalPoints])
    self.walkablePaths = np.unique(np.concatenate([self.verticalPaths, self.horizontalPaths]))
    self.multiPaths = np.concatenate([[s2c(l) for l in ['+', 'e']], globalPoints])

    self.spikeColorPairMasks = np.copy(self.squareColors)

    for geoDefinition in globalGeoStore:
      if geoDefinition[1] == 0:
        continue

      spikeIndex = np.argwhere(globalSpikes == s2c(geoDefinition[1]))[0][0]
      self.spikeColorPairMasks[spikeIndex] = np.any([self.spikeColorPairMasks[spikeIndex], self.onlyGeos == s2c(geoDefinition[0])], 0)

  # todo To support mishaps, just convert the mishap to a new spike with all other items
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
      np.all([shift(hMask, (1,), (0,), 0), hToP != s2c('p'), np.isin(self.starting, self.verticalPaths)], 0),
      np.all([shift(hMask, (1,), (1,), 0), hToP != s2c('p'), np.isin(self.starting, self.horizontalPaths)], 0),
      np.all([shift(hMask, (-1,), (0,), 0), hToP != s2c('p'), np.isin(self.starting, self.verticalPaths)], 0),
      np.all([shift(hMask, (-1,), (1,), 0), hToP != s2c('p'), np.isin(self.starting, self.horizontalPaths)], 0)
    ]

    next2MultiHMasks = [
      np.all([shift(hMask, (2,), (0,), 0), hToP != s2c('p'), np.isin(self.starting, self.verticalPaths)], 0),
      np.all([shift(hMask, (2,), (1,), 0), hToP != s2c('p'), np.isin(self.starting, self.horizontalPaths)], 0),
      np.all([shift(hMask, (-2,), (0,), 0), hToP != s2c('p'), np.isin(self.starting, self.verticalPaths)], 0),
      np.all([shift(hMask, (-2,), (1,), 0), hToP != s2c('p'), np.isin(self.starting, self.horizontalPaths)], 0)
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
    pathMask = np.any([current == s2c('h'), current == s2c('p')], 0)
    if not np.any(np.all([pathMask, self.esMask], 0)):
      return False
    
    if np.count_nonzero(np.all([np.any(self.points, 0), ~pathMask], 0)) > 0:
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
    walkablePaths = np.all([~pathMask, np.isin(self.starting, self.walkablePaths)], 0)
    nextSteps = np.any([shift(headMask, (1,), (0,), 0),shift(headMask, (-1,), (0,), 0), shift(headMask, (1,), (1,), 0), shift(headMask, (-1,), (1,), 0)], 0)
    if not np.any(np.all([nextSteps, walkablePaths], 0)):
      return True

    zoneIndices = self.getZoneIndices(current)
    headZoneIndices = self.fullMapToZoneIndex(np.argwhere(current == s2c('h')), zoneIndices)
    endZoneIndices = self.fullMapToZoneIndex(self.es, zoneIndices)

    # If the head can't reach any end
    if np.intersect1d(endZoneIndices, headZoneIndices).size == 0:
      return True
    
    # todo out of reach can be computed better with non reachable zones, making it a half closed zone
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
    
  # todo check if there are points left in the zone
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

      positiveAngledGeos = [z[1][0] for z in zonedGeos if not z[0]]
      positiveCount = np.sum([np.count_nonzero(z) for z in positiveAngledGeos])
      negativeAngledGeos = [z[1][0] for z in zonedGeos if z[0]]
      negativeCount = np.sum([np.count_nonzero(z) for z in negativeAngledGeos])

      if np.count_nonzero(zoneMask) != positiveCount - negativeCount:
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
      
      # todo a pending zone also has a minimum size, the 1d bordering line far from the puzzle border
      
    return False
  
  def recursiveCanPlaceAllGeos(self, zoneMaskLeft, geosLeftToPlace):
    if len(geosLeftToPlace) == 1:
      return len(self.getConfigurations(zoneMaskLeft, geosLeftToPlace[0][1], earlyReturn=True)) > 0
    
    geoConfigurations = []

    negativeGeosLeftToPlace = [g[1] for g in geosLeftToPlace if g[0]]
    positiveGeosLeftToPlace = [g for g in geosLeftToPlace if not g[0]]

    if len(negativeGeosLeftToPlace) > 0:
      for geo in negativeGeosLeftToPlace:
        configurations = self.getNegativeConfigurations(zoneMaskLeft, geo)
        if len(configurations) == 0:
          return False
        
        geoConfigurations.append((configurations, geo))

      geoConfigurations.sort(key=lambda x: len(x[0]))
      newRemainingGeos = [(True, x[1]) for x in geoConfigurations[1:]]

      for paddedZoneMask, geoConfiguration in geoConfigurations[0][0]:
        newZoneMask = np.any([paddedZoneMask, geoConfiguration], 0)
        if self.recursiveCanPlaceAllGeos(newZoneMask, newRemainingGeos + positiveGeosLeftToPlace):
          return True
        
      return False

    for _positive, geo in positiveGeosLeftToPlace:
      configurations = self.getConfigurations(zoneMaskLeft, geo, earlyReturn=False)
      if len(configurations) == 0:
        return False
      
      geoConfigurations.append((configurations, geo))

    geoConfigurations.sort(key=lambda x: len(x[0]))

    newRemainingGeos = [(False, x[1]) for x in geoConfigurations[1:]]
    
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

      paddedAngledGeo = pad(angledGeo, (zoneMask.shape[0] - angledGeo.shape[0], zoneMask.shape[1] - angledGeo.shape[1]), (0, 1), False)
      
      for topY in range(topZoneMask, bottomZoneMask - geoHeight + 1):
        for leftX in range(leftZoneMask, rightZoneMask - geoWidth + 1):
          shiftedAngledGeo = shift(paddedAngledGeo, (topY, leftX), (0, 1), 0)
          if np.all(np.all([shiftedAngledGeo, zoneMask], 0) == shiftedAngledGeo):
            configurations.append(shiftedAngledGeo)

            if earlyReturn:
              return np.array(configurations, dtype=np.bool)

    return np.array(configurations, dtype=np.bool)
  
  def getNegativeConfigurations(self, zoneMask, geo):
    configurations = []

    for angledGeo in geo:
      whereGeo = np.argwhere(angledGeo)

      topGeo = np.min(whereGeo[:,0])
      bottomGeo = np.max(whereGeo[:,0]) + 1
      leftGeo = np.min(whereGeo[:,1])
      rightGeo = np.max(whereGeo[:,1]) + 1
      geoHeight = bottomGeo - topGeo
      geoWidth = rightGeo - leftGeo
      
      paddedZoneMask = prepad(pad(zoneMask, (geoHeight, geoWidth), (0, 1), False), (geoHeight, geoWidth), (0, 1), False)

      oneAwayFromZoneMask = np.any([
        shift(paddedZoneMask, (1, 0), (0, 1), 0),
        shift(paddedZoneMask, (-1, 0), (0, 1), 0),
        shift(paddedZoneMask, (0, 1), (0, 1), 0),
        shift(paddedZoneMask, (0, -1), (0, 1), 0),
        paddedZoneMask
      ], 0)

      driftedZoneMask = oneAwayFromZoneMask
      for _yDrift in range(geoHeight - 1):
        driftedZoneMask = np.any([
          shift(driftedZoneMask, (-1, 0), (0, 1), 0),
          shift(driftedZoneMask, (0, -1), (0, 1), 0),
          driftedZoneMask
        ], 0)

      for _xDrift in range(geoWidth - 1):
        driftedZoneMask = np.any([
          shift(driftedZoneMask, (-1, 0), (0, 1), 0),
          shift(driftedZoneMask, (0, -1), (0, 1), 0),
          driftedZoneMask
        ], 0)

      whereDriftedZoneMask = np.argwhere(driftedZoneMask)
      topDriftedZoneMask = np.min(whereDriftedZoneMask[:,0])
      bottomDriftedZoneMask = np.max(whereDriftedZoneMask[:,0]) + 1
      leftDriftedZoneMask = np.min(whereDriftedZoneMask[:,1])
      rightDriftedZoneMask = np.max(whereDriftedZoneMask[:,1]) + 1
      driftedZoneMaskHeight = bottomDriftedZoneMask - topDriftedZoneMask
      driftedZoneMaskWidth = rightDriftedZoneMask - leftDriftedZoneMask

      if driftedZoneMaskHeight < geoHeight or driftedZoneMaskWidth < geoWidth:
        continue

      paddedAngledGeo = pad(angledGeo, (paddedZoneMask.shape[0] - angledGeo.shape[0], paddedZoneMask.shape[1] - angledGeo.shape[1]), (0, 1), False)
      
      for topY in range(topDriftedZoneMask, bottomDriftedZoneMask - geoHeight + 1):
        for leftX in range(leftDriftedZoneMask, rightDriftedZoneMask - geoWidth + 1):
          shiftedAngledGeo = shift(paddedAngledGeo, (topY, leftX), (0, 1), 0)
          if np.any(np.all([shiftedAngledGeo, oneAwayFromZoneMask], 0)) and not np.any(np.all([shiftedAngledGeo, paddedZoneMask], 0)):
            configurations.append([paddedZoneMask, shiftedAngledGeo])

    return np.array(configurations, dtype=np.bool)


  def getSmallNexts(self, current):
    hMask = current == s2c('h')
    hToP = np.where(hMask, s2c('p'), current)
    nextStates = []

    nextMultiHMasks = [
      np.all([shift(hMask, (1,), (0,), 0), hToP != s2c('p'), np.isin(self.starting, self.multiPaths)], 0),
      np.all([shift(hMask, (1,), (1,), 0), hToP != s2c('p'), np.isin(self.starting, self.multiPaths)], 0),
      np.all([shift(hMask, (-1,), (0,), 0), hToP != s2c('p'), np.isin(self.starting, self.multiPaths)], 0),
      np.all([shift(hMask, (-1,), (1,), 0), hToP != s2c('p'), np.isin(self.starting, self.multiPaths)], 0)
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
  
  def buildGeo(self, geoDefinition):
    (_char, _pairedSpike, rotate, _negative, array) = geoDefinition

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
    squareShape = (
      np.max([squareShape[0]] + [s.shape[0] for s in smallArrays]),
      np.max([squareShape[1]] + [s.shape[1] for s in smallArrays])
    )
    smallArrays = [s for s in smallArrays if s.shape[0] <= squareShape[0] and s.shape[1] <= squareShape[1]]

    bigArrays = np.array([np.concatenate([np.concatenate([x, np.zeros((squareShape[0] - x.shape[0], x.shape[1]))], 0), np.zeros((squareShape[0], squareShape[1] - x.shape[1]))], 1) for x in smallArrays])

    keptBigArrays = np.ones((bigArrays.shape[0]), dtype=np.bool)
    
    for dupeCandidateIndex in range(1, len(keptBigArrays)):
      for previousCandidate in range(0, dupeCandidateIndex):
        if np.all(bigArrays[dupeCandidateIndex] == bigArrays[previousCandidate]):
          keptBigArrays[dupeCandidateIndex] = False
          break

    return bigArrays[keptBigArrays]
  
# map letter, linked spike, rotatable, negative, matrix definition
globalGeoStore = [
  (
    'a',
    0,
    False,
    False,
    [
      [True],
      [True],
      [True],
      [True],
    ]
  ),(
    'b',
    0,
    False,
    False,
    [
      [True, True, True, True],
    ]
  )
]

globalSpikes = np.array([s2c(l) for l in ['z', 'y', 'x', 'w']])
globalSquares = np.array([s2c(l) for l in ['l', 'm', 'n', 'q']])
globalPoints = np.array([s2c(l) for l in ['t']])