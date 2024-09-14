from AbstractProblemDefinition import AbstractProblemDefinition
import numpy as np

from utils import s2c, shift, arrayToDebug

class ProblemDefinition(AbstractProblemDefinition):
  def __init__(self, starting: np.ndarray):
    self.starting = starting
    self.middleSquares = starting[slice(1,-1,2),slice(1,-1,2)]
    self.spikes = np.array([self.middleSquares == l for l in globalSpikes])
    self.squareColors = np.array([self.middleSquares == l for l in globalSquares])
    self.esMask = starting == s2c('e')
    self.es = np.argwhere(self.esMask)
    self.geos = {s2c(geo[0]): self.buildGeo(geo) for geo in globalGeoStore}
    self.onlyGeos = np.copy(self.middleSquares)
    self.onlyGeos[~np.isin(self.middleSquares, list(self.geos.keys()))] = 0
    self.orderedGeos = [(self.onlyGeos[a], a) for a in np.argwhere(self.onlyGeos != 0)]

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
    geoCount = np.count_nonzero(self.onlyGeos)
    groupCombinations = []
    for largestGroupSize in range(geoCount, 0, -1):
      groupCombinations += self.recursiveBuildGroupCombination(list(range(geoCount)), largestGroupSize)

    print("Possible combinations:", len(groupCombinations))

    possibleInclusiveGeos = []
    for groupCombination in groupCombinations:
      possibleInclusiveGeos += self.buildInclusiveGeos(groupCombination)

  def recursiveBuildGroupCombination(self, remainingToPlace, largestGroupSize):
    if largestGroupSize > len(remainingToPlace):
      return []
    
    if largestGroupSize == 1:
      return [[[i] for i in remainingToPlace]]
    
    groupsWithAtLeastOneOfSize = self.buildSingleGroups(largestGroupSize, remainingToPlace)

    groups = []

    for (partialGroup, outOfPartialGroup) in groupsWithAtLeastOneOfSize:
      if len(outOfPartialGroup) == 0:
        groups.append([partialGroup])
        continue

      for outOfPartialGroupSize in range(min(len(outOfPartialGroup), largestGroupSize), 0, -1):
        restOfCombinations = self.recursiveBuildGroupCombination(outOfPartialGroup, outOfPartialGroupSize)
        for restOfCombination in restOfCombinations:
          groups.append([partialGroup] + restOfCombination)

    return groups

  def buildSingleGroups(self, groupSize, items: list):
    if groupSize == 0:
      return [([], items)]
    
    if len(items) < groupSize:
      return []
    
    itemArray = np.array(items)
    selectionIndices = np.array(list(range(groupSize)))
    onesLikeItems = np.zeros_like(itemArray, dtype=np.bool)
    groups = []

    while len(items) - selectionIndices[0] > groupSize:
      # Add next group
      selectionMask = np.copy(onesLikeItems)
      selectionMask[selectionIndices] = True
      groups.append((itemArray[selectionMask].tolist(), itemArray[~selectionMask].tolist()))

      # Increase selection index
      for attemptIncrement in range(len(selectionIndices) - 1, -1, -1):
        if len(items) - selectionIndices[attemptIncrement] <= len(selectionIndices) - attemptIncrement:
          continue

        baseValue = selectionIndices[attemptIncrement]

        for counter, increment in enumerate(range(attemptIncrement, len(selectionIndices))):
          selectionIndices[increment] = baseValue + counter + 1

        break

    # Add last group
    selectionMask = np.copy(onesLikeItems)
    selectionMask[selectionIndices] = True
    groups.append((itemArray[selectionMask].tolist(), itemArray[~selectionMask].tolist()))

    return groups
  
  def buildInclusiveGeos(self, geoCombination):
    starting = self.recursiveBuildSingleInclusiveGeo(np.zeros_like(self.middleSquares, dtype=np.bool), np.ones_like(self.middleSquares, dtype=np.bool), np.zeros_like(self.middleSquares, dtype=np.bool), geoCombination[0], geoCombination[0])
    previous = []

    for geoCombinationIndex in range(1, len(geoCombination)):
      previouses = starting
      starting = []
      for previous in previouses:
        mustNotTouch = np.all([np.any([
          previous,
          shift(previous, (1, 0), (0, 1), 0),
          shift(previous, (0, 1), (0, 1), 0),
          shift(previous, (0, -1), (0, 1), 0),
          shift(previous, (-1, 0), (0, 1), 0)
        ], 0)])

        if np.all(mustNotTouch):
          continue

        newPartialInclusives = self.recursiveBuildSingleInclusiveGeo(np.zeros_like(self.middleSquares, dtype=np.bool), np.ones_like(self.middleSquares, dtype=np.bool), mustNotTouch, geoCombination[geoCombinationIndex], geoCombination[geoCombinationIndex])

        for newPartialInclusive in newPartialInclusives:
          starting.append(np.any([previous, newPartialInclusive], 0))

    return starting
    
  def recursiveBuildSingleInclusiveGeo(self, partialInclusiveGeo, mustTouchOneOf, mustNotTouch, remainingGeosToPlace, remainingGeosToTouch):
    if not np.any(np.all([mustTouchOneOf, ~mustNotTouch], 0)):
      return []
    
    if len(remainingGeosToPlace) > np.count_nonzero(remainingGeosToTouch):
      return []
    
    if len(remainingGeosToPlace) == 0:
      return [partialInclusiveGeo]
    
    inclusiveGeos = []
    attemptedGeo = set()

    for geoToPlaceIndex in range(len(remainingGeosToPlace)):
      remainingGeosToPlaceCopy = list(remainingGeosToPlace)
      geoToPlace = remainingGeosToPlaceCopy[geoToPlaceIndex]
      del remainingGeosToPlaceCopy[geoToPlaceIndex]

      if self.orderedGeos[geoToPlace][0] in attemptedGeo:
        continue

      attemptedGeo.add(self.orderedGeos[geoToPlace])

      placements = self.getGeoPlacements(geoToPlace, mustTouchOneOf, mustNotTouch, remainingGeosToTouch)

      for placement in placements:
        newPartialInclusiveGeo = np.any([partialInclusiveGeo, placement], 0)
        newMustNotTouch = np.any([mustNotTouch, newPartialInclusiveGeo], 0)
        newMustTouchOneOf = np.all([np.any([
          shift(newPartialInclusiveGeo, (1, 0), (0, 1), 0),
          shift(newPartialInclusiveGeo, (0, 1), (0, 1), 0),
          shift(newPartialInclusiveGeo, (0, -1), (0, 1), 0),
          shift(newPartialInclusiveGeo, (-1, 0), (0, 1), 0)
        ], 0), ~newMustNotTouch])

        if not np.any(newMustTouchOneOf):
          continue

        newRemainingGeosToTouch = np.all([remainingGeosToTouch, ~newPartialInclusiveGeo])

        inclusiveGeos += self.recursiveBuildSingleInclusiveGeo(newPartialInclusiveGeo, newMustTouchOneOf, newMustNotTouch, remainingGeosToPlaceCopy, newRemainingGeosToTouch)

  def getGeoPlacements(self, geoToPlace, mustTouchOneOf, mustNotTouch, remainingGeosToTouch):
    geos = self.geos[self.orderedGeos[geoToPlace][1]]
    todo [(xMin, xMax) for geo in geos for geoToTouch in remainingGeosToTouch]
    xBounds = [() for geo in geos]
    yBounds = [() for geo in geos]
    placements = []
    for geoIndex in range(len(geos)):
      for xDisplacement in range(xBounds[geoIndex][0], xBounds[geoIndex][1], 1):
        for yDisplacement in range(yBounds[geoIndex][0], yBounds[geoIndex][1], 1):
          tentativeGeoPlacement = shift(geos[geoIndex], (xDisplacement, yDisplacement), (0, 1), 0)

          if np.any(np.all([tentativeGeoPlacement, mustNotTouch], 0)):
            continue

          if not np.any(np.all([mustTouchOneOf], 0)):
            continue

          if np.max(remainingGeosToTouch) != 0 and not np.any(np.all([tentativeGeoPlacement, remainingGeosToTouch], 0)):
            continue

          placements.append(tentativeGeoPlacement)

    unicite placements before continuing

    return placements

  def getNexts(self, current):
    pass

  def isSatisfied(self, current):
    pass
  
  def isUnsatisfiable(self, current):
    pass
  
  def evalRemaining(self, current):
    pass
  
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

    bigArrays = np.array([np.concatenate([np.concatenate([x, np.zeros((squareShape[0] - x.shape[0], x.shape[1]))], 0), np.zeros((squareShape[0], squareShape[1] - x.shape[1]))], 1) for x in smallArrays])

    keptBigArrays = np.ones((bigArrays.shape[0]), dtype=np.bool)
    
    for dupeCandidateIndex in range(1, len(keptBigArrays)):
      for previousCandidate in range(0, dupeCandidateIndex):
        if np.all(bigArrays[dupeCandidateIndex] == bigArrays[previousCandidate]):
          keptBigArrays[dupeCandidateIndex] = False
          break

    return bigArrays[keptBigArrays]
  
# map letter, linked spike, rotatable, matrix definition
globalGeoStore = [
  (
    'a',
    0,
    False,
    [
      [True],
      [True],
    ]
  )
]

globalSpikes = np.array([s2c(l) for l in ['z', 'y', 'x']])
globalSquares = np.array([s2c(l) for l in ['l', 'm', 'n']])