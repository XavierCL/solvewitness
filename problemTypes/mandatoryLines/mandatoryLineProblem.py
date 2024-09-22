from AbstractProblemDefinition import AbstractProblemDefinition
import numpy as np

from utils import s2c, shift, arrayToDebug

class ProblemDefinition(AbstractProblemDefinition):
  def __init__(self, starting: np.ndarray):
    self.starting = starting
    self.middleSquares = starting[slice(1,-1,2),slice(1,-1,2)]

    self.edgeMiddleMask = np.zeros_like(self.middleSquares, dtype=np.bool)
    self.edgeMiddleMask = np.any([
      shift(self.edgeMiddleMask, (1, 1), (0, 1), 1),
      shift(self.edgeMiddleMask, (-1, -1), (0, 1), 1)
    ], 0)

    self.edgeMask = np.zeros_like(self.starting, dtype=np.bool)
    self.edgeMask = np.any([
      shift(self.edgeMask, (1, 1), (0, 1), 1),
      shift(self.edgeMask, (-1, -1), (0, 1), 1)
    ], 0)

    self.spikes = np.array([self.middleSquares == l for l in globalSpikes])
    self.squareColors = np.array([self.middleSquares == l for l in globalSquares])
    self.points = np.array([starting == l for l in globalPoints])
    self.esMask = starting == s2c('e')
    self.es = np.argwhere(self.esMask)
    self.geos = {s2c(geo[0]): self.buildGeo(geo) for geo in globalGeoStore}
    self.onlyGeos = np.copy(self.middleSquares)
    self.onlyGeos[~np.isin(self.middleSquares, list(self.geos.keys()))] = 0
    self.orderedGeos = [(self.onlyGeos[tuple(a)], a) for a in np.argwhere(self.onlyGeos != 0)]
    self.verticalPaths = np.concatenate([[s2c(l) for l in ['+','|','e','s']], globalPoints])
    self.verticalPathMask = np.isin(self.starting, self.verticalPaths)
    self.horizontalPaths = np.concatenate([[s2c(l) for l in ['+','-','e','s']], globalPoints])
    self.horizontalPathsMask = np.isin(self.starting, self.horizontalPaths)
    self.walkablePaths = np.unique(np.concatenate([self.verticalPaths, self.horizontalPaths]))
    multiPaths = np.concatenate([[s2c(l) for l in ['+', 'e']], globalPoints])
    self.multiPathMask = np.isin(self.starting, multiPaths)
    self.walkableMask = np.isin(self.starting, self.walkablePaths)

    self.spikeColorPairMasks = np.copy(self.squareColors)

    for geoDefinition in globalGeoStore:
      if geoDefinition[1] == 0:
        continue

      spikeIndex = np.argwhere(globalSpikes == s2c(geoDefinition[1]))[0][0]
      self.spikeColorPairMasks[spikeIndex] = np.any([self.spikeColorPairMasks[spikeIndex], self.onlyGeos == s2c(geoDefinition[0])], 0)

    # Build group combinations
    geoCount = np.count_nonzero(self.onlyGeos)
    groupCombinations = []
    for largestGroupSize in range(geoCount, 0, -1):
      groupCombinations += self.recursiveBuildGroupCombination(list(range(geoCount)), largestGroupSize)

    print("Possible combinations:", len(groupCombinations))

    # Build geo mandatory squares
    possibleInclusiveGeos = []
    for groupCombination in groupCombinations:
      possibleInclusiveGeos += self.buildInclusiveGeos(groupCombination)

    print("Geo combinations:", len(possibleInclusiveGeos))

    # Build geo mandatory lines
    unvisitedGeoMandatoryLines = [self.inclusiveGeoToMandatoryLines(o) for o in possibleInclusiveGeos]
    unvisitedGeoMandatoryLines = np.array([o for o in unvisitedGeoMandatoryLines if o is not None])

    print("Single lined geo combinations:", len(unvisitedGeoMandatoryLines))

    self.geoMandatoryLines = [self.visitMandatoryLines(o) for o in unvisitedGeoMandatoryLines]
    self.geoMandatoryLines = [o for o in self.geoMandatoryLines if o is not None]
    self.geoMandatoryLines.sort(key=lambda x: np.count_nonzero(~x[1]) - np.count_nonzero(x[0]))
    self.geoMandatoryLines = np.array(self.geoMandatoryLines)

    print("Mandatory geo lines:", len(self.geoMandatoryLines))
    print("Uncertain paths: ", [np.count_nonzero(~x[1]) - np.count_nonzero(x[0]) for x in self.geoMandatoryLines])
    print('\n'.join(arrayToDebug(np.where(self.geoMandatoryLines[0,0], s2c('p'), 0))))

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
    remainingGeosToTouch = np.zeros_like(self.middleSquares, dtype=np.bool)
    remainingGeosToTouch[tuple(np.array([self.orderedGeos[o][1] for o in geoCombination[0]]).T)] = 1
    starting = [[o] for o in self.recursiveBuildSingleInclusiveGeo(np.zeros_like(self.middleSquares, dtype=np.bool), np.ones_like(self.middleSquares, dtype=np.bool), np.zeros_like(self.middleSquares, dtype=np.bool), geoCombination[0], remainingGeosToTouch)]
    previouses = []

    for geoCombinationIndex in range(1, len(geoCombination)):
      previouses = starting
      starting = []
      for previousInclusives in previouses:
        oredPrevious = np.any(previousInclusives, 0)
        remainingGeosToTouch = np.zeros_like(self.middleSquares, dtype=np.bool)
        remainingGeosToTouch[tuple(np.array([self.orderedGeos[o][1] for o in geoCombination[geoCombinationIndex]]).T)] = 1
        newPartialInclusives = self.recursiveBuildSingleInclusiveGeo(np.zeros_like(self.middleSquares, dtype=np.bool), np.ones_like(self.middleSquares, dtype=np.bool), oredPrevious, geoCombination[geoCombinationIndex], remainingGeosToTouch)

        for newPartialInclusive in newPartialInclusives:
          starting.append(previousInclusives + [newPartialInclusive])

      # reduce duplicates
      for s in starting:
        # Normalize partial inclusives
        s.sort(key=lambda x: np.argwhere(x.reshape(-1))[0,0])
      nextStartings = []
      startingSet = set()
      for start in starting:
        startingTuple = tuple(np.reshape(start, -1))
        if startingTuple not in startingSet:
          startingSet.add(startingTuple)
          nextStartings.append(start)

      starting = nextStartings

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

      attemptedGeo.add(self.orderedGeos[geoToPlace][0])

      placements = self.getGeoPlacements(geoToPlace, mustTouchOneOf, mustNotTouch, remainingGeosToTouch, len(remainingGeosToPlaceCopy) == 0)

      for placement in placements:
        newPartialInclusiveGeo = np.any([partialInclusiveGeo, placement], 0)
        newMustNotTouch = np.any([mustNotTouch, newPartialInclusiveGeo], 0)
        newMustTouchOneOf = np.all([np.any([
          shift(newPartialInclusiveGeo, (1, 0), (0, 1), 0),
          shift(newPartialInclusiveGeo, (0, 1), (0, 1), 0),
          shift(newPartialInclusiveGeo, (0, -1), (0, 1), 0),
          shift(newPartialInclusiveGeo, (-1, 0), (0, 1), 0)
        ], 0), ~newMustNotTouch], 0)

        if not np.any(newMustTouchOneOf):
          continue

        newRemainingGeosToTouch = np.all([remainingGeosToTouch, ~newPartialInclusiveGeo], 0)

        inclusiveGeos += self.recursiveBuildSingleInclusiveGeo(newPartialInclusiveGeo, newMustTouchOneOf, newMustNotTouch, remainingGeosToPlaceCopy, newRemainingGeosToTouch)

    # Reduce duplicates
    dedupedInclusiveGeos = []
    inclusiveSet = set()
    for inclusiveGeo in inclusiveGeos:
      inclusiveTuple = tuple(inclusiveGeo.reshape(-1))
      if inclusiveTuple not in inclusiveSet:
        inclusiveSet.add(inclusiveTuple)
        dedupedInclusiveGeos.append(inclusiveGeo)

    return dedupedInclusiveGeos

  def getGeoPlacements(self, geoToPlace, siblingPlacements, mustNotTouch, remainingGeosToTouch, mustTouchEdge):
    geos = self.geos[self.orderedGeos[geoToPlace][0]]
    mustTouchOneOfs = [siblingPlacements, remainingGeosToTouch]
    if mustTouchEdge:
      mustTouchOneOfs.append(self.edgeMiddleMask)
    mustTouchOneOfs = np.array(mustTouchOneOfs)
    yMustTouches = np.any(mustTouchOneOfs, axis=2)
    xMustTouches = np.any(mustTouchOneOfs, axis=1)
    yInclusiveMustTouches = np.all([np.cumsum(yMustTouches, 1), np.flip(np.cumsum(np.flip(yMustTouches, 1), 1), 1)], 1)
    xInclusiveMustTouches = np.all([np.cumsum(xMustTouches, 1), np.flip(np.cumsum(np.flip(xMustTouches, 1), 1), 1)], 1)
    maxY = np.argwhere(yInclusiveMustTouches[0]).reshape(-1)[0]
    minY = np.argwhere(yInclusiveMustTouches[1]).reshape(-1)[-1]
    maxX = np.argwhere(xInclusiveMustTouches[0]).reshape(-1)[0]
    minX = np.argwhere(xInclusiveMustTouches[1]).reshape(-1)[-1]
    whereGeos = [np.argwhere(geo) for geo in geos]
    geoHeights = [geo[:,0].max() - geo[:,0].min() + 1 for geo in whereGeos]
    geoWidths = [geo[:,1].max() - geo[:,1].min() + 1 for geo in whereGeos]
    yBounds = [(max(0, maxY - geoHeights[geoIndex] + 1), min(self.middleSquares.shape[0] - geoHeights[geoIndex] + 1, minY + 1)) for geoIndex in range(len(geos))]
    xBounds = [(max(0, maxX - geoWidths[geoIndex] + 1), min(self.middleSquares.shape[1] - geoWidths[geoIndex] + 1, minX + 1)) for geoIndex in range(len(geos))]
    placements = []
    for geoIndex in range(len(geos)):
      for yDisplacement in range(yBounds[geoIndex][0], yBounds[geoIndex][1], 1):
        for xDisplacement in range(xBounds[geoIndex][0], xBounds[geoIndex][1], 1):
          tentativeGeoPlacement = shift(geos[geoIndex], (yDisplacement, xDisplacement), (0, 1), 0)

          if np.any(np.all([tentativeGeoPlacement, mustNotTouch], 0)):
            continue

          if not np.all(np.any(np.all([mustTouchOneOfs, np.repeat(tentativeGeoPlacement[np.newaxis,:,:], mustTouchOneOfs.shape[0], 0)], 0), (1, 2))):
            continue

          placements.append(tentativeGeoPlacement)

    return placements
  
  # returns an np array on the starting map or None if impossible
  def inclusiveGeoToMandatoryLines(self, inclusiveGeos):
    mandatoryLines = np.zeros_like(self.starting, np.bool)
    forbiddenLines = np.zeros_like(self.starting, np.bool)
    for inclusiveGeo in inclusiveGeos:
      geoMaskIndices = tuple(np.array(np.where(inclusiveGeo)) * 2 + 1)
      middleGeoLines = np.zeros_like(self.starting, np.bool)
      middleGeoLines[geoMaskIndices] = True
      attemptGeoLines = np.all([np.any([
        np.all([shift(middleGeoLines, (1, 0), (0, 1), 0), ~shift(middleGeoLines, (-1, 0), (0, 1), 0)], 0),
        np.all([shift(middleGeoLines, (-1, 0), (0, 1), 0), ~shift(middleGeoLines, (1, 0), (0, 1), 0)], 0),
        np.all([shift(middleGeoLines, (0, 1), (0, 1), 0), ~shift(middleGeoLines, (0, -1), (0, 1), 0)], 0),
        np.all([shift(middleGeoLines, (0, -1), (0, 1), 0), ~shift(middleGeoLines, (0, 1), (0, 1), 0)], 0),
        np.all([shift(middleGeoLines, (1, 1), (0, 1), 0), ~shift(middleGeoLines, (-1, -1), (0, 1), 0)], 0),
        np.all([shift(middleGeoLines, (1, -1), (0, 1), 0), ~shift(middleGeoLines, (-1, 1), (0, 1), 0)], 0),
        np.all([shift(middleGeoLines, (-1, 1), (0, 1), 0), ~shift(middleGeoLines, (1, -1), (0, 1), 0)], 0),
        np.all([shift(middleGeoLines, (-1, -1), (0, 1), 0), ~shift(middleGeoLines, (1, 1), (0, 1), 0)], 0),
      ], 0), ~self.edgeMask], 0)

      # Add the multipath on the edge of the geo line
      attemptGeoLines = np.any([attemptGeoLines, np.all([np.any([
        shift(attemptGeoLines, (1, 0), (0, 1), 0),
        shift(attemptGeoLines, (-1, 0), (0, 1), 0),
        shift(attemptGeoLines, (0, 1), (0, 1), 0),
        shift(attemptGeoLines, (0, -1), (0, 1), 0),
      ], 0), self.edgeMask], 0)], 0)

      if np.any(np.all([attemptGeoLines, ~self.walkableMask], 0)):
        return None
      
      attemptForbiddenLines = np.all([np.any([
        shift(middleGeoLines, (1, 0), (0, 1), 0),
        shift(middleGeoLines, (0, 1), (0, 1), 0),
      ], 0), ~self.edgeMask, ~attemptGeoLines], 0)
      
      mandatoryLines = np.any([mandatoryLines, attemptGeoLines], 0)
      forbiddenLines = np.any([forbiddenLines, attemptForbiddenLines], 0)

    # Remove triple and quadruble links
    multiPaths = np.zeros_like(self.starting, dtype=np.bool)
    multiPaths[slice(0, multiPaths.shape[0], 2), slice(0, multiPaths.shape[1], 2)] = True
    shiftedMandatoryMultiPaths = np.all([[
      shift(multiPaths, (1, 0), (0, 1), 0),
      shift(multiPaths, (-1, 0), (0, 1), 0),
      shift(multiPaths, (0, 1), (0, 1), 0),
      shift(multiPaths, (0, -1), (0, 1), 0),
    ], np.repeat(mandatoryLines[np.newaxis,:,:], 4, 0)], 0)
    multiPathSiblings = np.count_nonzero([
      shift(shiftedMandatoryMultiPaths[0], (-1, 0), (0, 1), 0),
      shift(shiftedMandatoryMultiPaths[1], (1, 0), (0, 1), 0),
      shift(shiftedMandatoryMultiPaths[2], (0, -1), (0, 1), 0),
      shift(shiftedMandatoryMultiPaths[3], (0, 1), (0, 1), 0),
    ], axis=0)
    
    if np.any(multiPathSiblings > 2):
      return None

    return (mandatoryLines, forbiddenLines)
  
  def visitMandatoryLines(self, mandatoryAndForbidden):
    mandatoryLines = mandatoryAndForbidden[0]
    forbiddenLines = mandatoryAndForbidden[1]
    successfulVisits = []
    headObjectives = self.getOrderedHeadObjectives(mandatoryLines)
    endObjectives = self.es

    for startIndex in np.argwhere(self.starting == s2c('s')):
      for objectiveOrder in self.buildObjectiveOrders(len(headObjectives)):
        startObjectivePosition = startIndex
        mandatoryObjective = np.zeros_like(mandatoryLines)
        mandatoryObjective[tuple(startObjectivePosition)] = True
        anyVisit = np.zeros_like(forbiddenLines)
        anyVisit[tuple(startObjectivePosition)] = True
        successfulObjectives = 0
        for objectiveIndex in range(0, len(objectiveOrder), 2):
          flood = self.floodUsefulPath(startObjectivePosition, headObjectives[objectiveOrder[objectiveIndex]], np.any([forbiddenLines, mandatoryLines, mandatoryObjective], 0))

          if flood is None:
            break

          (mandatoryFlood, usefulPaths) = flood
          anyVisit = np.any([anyVisit, usefulPaths], 0)
          mandatoryObjective = np.any([mandatoryObjective, mandatoryFlood], 0)
          startObjectivePosition = headObjectives[objectiveOrder[objectiveIndex + 1]]
          successfulObjectives += 2

        if successfulObjectives < len(objectiveOrder):
          continue

        for endObjective in endObjectives:
          flood = self.floodUsefulPath(startObjectivePosition, endObjective, np.any([forbiddenLines, mandatoryLines, mandatoryObjective], 0))
          if flood is None:
            continue

          (mandatoryFlood, usefulPaths) = flood
          endAnyVisit = np.any([mandatoryLines, anyVisit, usefulPaths], 0)
          endMandatoryObjective = np.any([mandatoryLines, mandatoryObjective, mandatoryFlood], 0)

          successfulVisits.append((endMandatoryObjective, endAnyVisit))

    if len(successfulVisits) == 0:
      return None
    
    newForbiddenLines = ~np.any([s[1] for s in successfulVisits], 0)
    newMandatoryLines = np.all([s[0] for s in successfulVisits], 0)

    return [newMandatoryLines, newForbiddenLines]
  
  def getOrderedHeadObjectives(self, mandatoryLines):
    unsatisfiedHeads = np.zeros_like(self.starting, np.bool)
    unsatisfiedHeads[slice(0, self.starting.shape[0], 2), slice(0, self.starting.shape[1], 2)] = True
    unsatisfiedHeads = np.all([unsatisfiedHeads, self.edgeMask, mandatoryLines], 0)
    unsatisfiedHeadIndices = np.argwhere(unsatisfiedHeads)
    accountedHeads = set()
    orderedHeads = []
    for unsatisfiedHeadIndex in unsatisfiedHeadIndices:
      if tuple(unsatisfiedHeadIndex) in accountedHeads:
        continue

      # Find the pair head
      reachables = self.floodReachable(unsatisfiedHeadIndex, unsatisfiedHeadIndex, ~mandatoryLines)
      reachableHeads = np.all([reachables, unsatisfiedHeads], 0)
      reachableHeads[tuple(unsatisfiedHeadIndex)] = False
      otherHeadIndex = np.argwhere(reachableHeads)[0]
      accountedHeads.add(tuple(unsatisfiedHeadIndex))
      accountedHeads.add(tuple(otherHeadIndex))
      orderedHeads.append(unsatisfiedHeadIndex)
      orderedHeads.append(otherHeadIndex)

    return orderedHeads
  
  def buildObjectiveOrders(self, objectiveCount):
    # Takes a list of numbers.
    # Each pair of number in the input order are 'Pairs' of objectives.
    # Pairs must always stick next to another in the output
    # Outputs the 2d array of all possible orders for those objectives
    semiOrders = self.recursiveBuildOrders(list(range(int(objectiveCount / 2))))
    objectiveSwitches = self.buildAllBinary(int(objectiveCount / 2))

    orders = []
    for semiOrder in semiOrders:
      for objectiveSwitch in objectiveSwitches:
        manglingOrder = [o for s in semiOrder for o in [s * 2, s * 2 + 1]]
        for switchIndex, switch in enumerate(objectiveSwitch):
          if switch:
            (manglingOrder[switchIndex * 2], manglingOrder[switchIndex * 2 + 1]) = (manglingOrder[switchIndex * 2 + 1], manglingOrder[switchIndex * 2])
        orders.append(manglingOrder)

    return orders

  def buildAllBinary(self, count):
    if count == 0:
      return []
    
    switches = np.zeros(count, dtype=np.bool)
    binaries = []

    while len(binaries) != 2**count:
      binaries.append(np.copy(switches))
      for switchIndex in range(-1, -count - 1, -1):
        switches[switchIndex] = ~switches[switchIndex]
        if switches[switchIndex]:
          break
      
    return binaries

  def recursiveBuildOrders(self, remainingItems: list) -> list[list[int]]:
    if len(remainingItems) < 2:
      return [remainingItems]
    
    orders = []
    for placingItemIndex, placingItem in enumerate(remainingItems):
      newRemainingItems = remainingItems.copy()
      del newRemainingItems[placingItemIndex]
      for nextLists in self.recursiveBuildOrders(newRemainingItems):
        orders.append([placingItem] + nextLists)
    return orders
  
  def floodUsefulPath(self, startPos, endPos, forbiddenLines):
    forbiddenLines = np.any([forbiddenLines, ~self.walkableMask], 0)
    aPath = self.floodSinglePath(startPos, endPos, forbiddenLines)
    if aPath is None:
      return None
    
    # Build mandatory path
    bridgePaths = np.all([aPath, ~self.multiPathMask], 0)
    bridgePaths[tuple(startPos)] = False
    bridgePaths[tuple(endPos)] = False
    mandatoryPath = np.zeros_like(forbiddenLines)
    for pathLine in np.argwhere(bridgePaths):
      if self.verticalPathMask[tuple(pathLine)]:
        start = pathLine + [1, 0]
        end = pathLine + [-1, 0]
      else:
        start = pathLine + [0, 1]
        end = pathLine + [0, -1]

      forbiddenPathLine = np.copy(forbiddenLines)
      forbiddenPathLine[tuple(pathLine)] = True

      if self.floodSinglePath(start, end, forbiddenPathLine, actualPath=False) is None:
        mandatoryPath[tuple(pathLine)] = True
        mandatoryPath[tuple(start)] = True
        mandatoryPath[tuple(end)] = True

    # Build useful paths
    singleWayLines = np.zeros_like(aPath)
    bridgesNotInSuccessfulPath = np.all([self.floodReachable(startPos, endPos, forbiddenLines), ~aPath, ~self.multiPathMask], 0)
    for potentialLine in np.argwhere(bridgesNotInSuccessfulPath):
      if self.verticalPathMask[tuple(potentialLine)]:
        start = potentialLine + [1, 0]
        end = potentialLine + [-1, 0]
      else:
        start = potentialLine + [0, 1]
        end = potentialLine + [0, -1]

      if forbiddenLines[tuple(start)] or forbiddenLines[tuple(end)]:
        singleWayLines[tuple(potentialLine)] = True
        continue
      
      forbiddenPathLine = np.copy(forbiddenLines)
      forbiddenPathLine[tuple(potentialLine)] = True

      if self.floodSinglePath(start, end, forbiddenPathLine, actualPath=False) is None:
        singleWayLines[tuple(potentialLine)] = True

    usefulPaths = self.floodReachable(startPos, endPos, np.any([forbiddenLines, singleWayLines], 0))
    usefulPaths[tuple(startPos)] = True
    usefulPaths[tuple(endPos)] = True

    return (mandatoryPath, usefulPaths)
  
  def floodSinglePath(self, startPos, endPos, forbiddenLines, actualPath=True):
    startMap = np.zeros_like(forbiddenLines, dtype=np.int32)
    startMap[tuple(startPos)] = 1
    endMap = np.zeros_like(forbiddenLines, dtype=np.int32)
    endMap[tuple(endPos)] = 1

    previousStartMap = np.zeros_like(forbiddenLines)
    previousEndMap = np.zeros_like(forbiddenLines)

    while np.any(startMap != previousStartMap) and np.any(endMap != previousEndMap):
      previousStartMap = startMap
      previousEndMap = endMap

      baseStartMap = np.where(startMap > 0, startMap + 1, 0)
      startMap = np.where(forbiddenLines, 0, np.where(startMap > 0, startMap, np.max([
        shift(baseStartMap, (1, 0), (0, 1), 0),
        shift(baseStartMap, (0, 1), (0, 1), 0),
        shift(baseStartMap, (0, -1), (0, 1), 0),
        shift(baseStartMap, (-1, 0), (0, 1), 0),
      ], 0)))

      baseEndMap = np.where(endMap > 0, endMap + 1, 0)
      endMap = np.where(forbiddenLines, 0, np.where(endMap > 0, endMap, np.max([
        shift(baseEndMap, (1, 0), (0, 1), 0),
        shift(baseEndMap, (0, 1), (0, 1), 0),
        shift(baseEndMap, (0, -1), (0, 1), 0),
        shift(baseEndMap, (-1, 0), (0, 1), 0),
      ], 0)))

      if np.any(np.all([startMap > 0, endMap > 0], 0)):
        if not actualPath:
          return True

        startMap[tuple(startPos)] = 1
        endMap[tuple(endPos)] = 1
        path = np.zeros_like(forbiddenLines)
        startCollisionIndex = np.argwhere(np.all([startMap > 0, endMap > 0], 0))[0]
        path[tuple(startCollisionIndex)] = True
        endCollisionIndex = startCollisionIndex

        while not path[tuple(startPos)]:
          currentStartPathLength = startMap[tuple(startCollisionIndex)]
          neighbourIndices = np.array([s for s in [
            startCollisionIndex + [1, 0],
            startCollisionIndex + [-1, 0],
            startCollisionIndex + [0, 1],
            startCollisionIndex + [0, -1]
          ] if np.all([s >= 0, s < np.array(self.starting.shape)])])
          startCollisionIndex = neighbourIndices[np.argwhere(startMap[tuple(neighbourIndices.T)] == currentStartPathLength - 1)[0, 0]]
          path[tuple(startCollisionIndex)] = True

        while not path[tuple(endPos)]:
          currentEndPathLength = endMap[tuple(endCollisionIndex)]
          neighbourIndices = np.array([s for s in [
            endCollisionIndex + [1, 0],
            endCollisionIndex + [-1, 0],
            endCollisionIndex + [0, 1],
            endCollisionIndex + [0, -1]
          ] if np.all([s >= 0, s < np.array(self.starting.shape)])])
          endCollisionIndex = neighbourIndices[np.argwhere(endMap[tuple(neighbourIndices.T)] == currentEndPathLength - 1)[0, 0]]
          path[tuple(endCollisionIndex)] = True
        
        return path

    return None

  def floodReachable(self, startPos, endPos, forbiddenLines) -> np.ndarray[np.bool]:
    startMap = np.zeros_like(forbiddenLines)
    startMap[tuple(startPos)] = 1
    startMap[tuple(endPos)] = 1

    previousStartMap = np.zeros_like(forbiddenLines)

    while np.any(startMap != previousStartMap):
      previousStartMap = startMap
      startMap = np.all([np.any([
        shift(startMap, (1, 0), (0, 1), 0),
        shift(startMap, (0, 1), (0, 1), 0),
        shift(startMap, (0, -1), (0, 1), 0),
        shift(startMap, (-1, 0), (0, 1), 0),
        startMap
      ], 0), ~forbiddenLines], 0)

    return startMap

  # To support mishaps, just convert the mishap to a new spike with all other items
  # Add neighbouring color squares to those mandatory lines
  # Create mandatory points on points
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
    candidateMandatoryLines = self.getValidMandatoryLines(current, self.geoMandatoryLines)
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

      newCandidateMandatoryLines = self.getValidMandatoryLines(nextState, candidateMandatoryLines)

      if len(newCandidateMandatoryLines) == 0:
        continue

      nextStates.append((nextState, newCandidateMandatoryLines))

    nextStates.sort(key=lambda x: ~np.all(candidateMandatoryLines[0] == x[1][0]))
    nextStates.sort(key=lambda x: self.evalRemaining(x[0]))
    nextStates = [x[0] for x in nextStates]

    return nextStates

  def isSatisfied(self, current):
    pathMask = np.any([current == s2c('h'), current == s2c('p')], 0)
    if not np.any(np.all([pathMask, self.esMask], 0)):
      return False
    
    if np.count_nonzero(np.all([np.any(self.points, 0), ~pathMask], 0)) > 0:
      return False
    
    partialMandatoryLines = self.getValidMandatoryLines(current, self.geoMandatoryLines)
    if ~np.any(np.all(np.all([partialMandatoryLines[:,0], np.repeat(pathMask[np.newaxis,:,:], partialMandatoryLines.shape[0], 0)], 0) == partialMandatoryLines[:,0], (1, 2))):
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
    
    if len(self.getValidMandatoryLines(current, self.geoMandatoryLines)) == 0:
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
    pAndH = np.isin(current, [s2c('p'), s2c('h')])
    firstCandidateMandatoryLine = self.getValidMandatoryLines(current, self.geoMandatoryLines)[0]
    hPos = np.argwhere(current == s2c('h'))[0]
    unsatisfiedHeads = np.zeros_like(self.starting, np.bool)
    unsatisfiedHeads[slice(0, self.starting.shape[0], 2), slice(0, self.starting.shape[1], 2)] = True
    unsatisfiedHeads = np.all([unsatisfiedHeads, pAndH, self.edgeMask, firstCandidateMandatoryLine[0]], 0)
    unsatisfiedHeadIndices = np.argwhere(unsatisfiedHeads)

    if len(unsatisfiedHeadIndices) == 0:
      return int(np.min(np.sum(np.abs(self.es - hPos), axis=1)))
    
    return int(np.min(np.sum(np.abs(unsatisfiedHeadIndices - hPos), axis=1)))
  
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
    
    # Geos handled solely at the mandatory line level
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
    
    # Geos are handled at the mandatory line level
      
    return False

  def getSmallNexts(self, current):
    hMask = current == s2c('h')
    hToP = np.where(hMask, s2c('p'), current)
    nextStates = []

    nextMultiHMasks = [
      np.all([shift(hMask, (1,), (0,), 0), hToP != s2c('p'), self.multiPathMask], 0),
      np.all([shift(hMask, (1,), (1,), 0), hToP != s2c('p'), self.multiPathMask], 0),
      np.all([shift(hMask, (-1,), (0,), 0), hToP != s2c('p'), self.multiPathMask], 0),
      np.all([shift(hMask, (-1,), (1,), 0), hToP != s2c('p'), self.multiPathMask], 0)
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
  
  def getValidMandatoryLines(self, current, mandatoryLines):
    hAndP = np.isin(current, [s2c('p'), s2c('h')])

    forbiddenSatisfied = mandatoryLines[~np.any(np.all([np.repeat(hAndP[np.newaxis,:,:], mandatoryLines.shape[0], 0), mandatoryLines[:,1]], 0), (1, 2))]

    # Remove triple and quadruble links
    multiPaths = np.zeros_like(self.starting, dtype=np.bool)
    multiPaths[slice(0, multiPaths.shape[0], 2), slice(0, multiPaths.shape[1], 2)] = True
    mandatoryAndVisited = np.any([np.repeat(hAndP[np.newaxis,:,:], forbiddenSatisfied.shape[0], 0), forbiddenSatisfied[:,0]], 0)
    shiftedMultiPaths = np.repeat(np.array([
      shift(multiPaths, (1, 0), (0, 1), 0),
      shift(multiPaths, (-1, 0), (0, 1), 0),
      shift(multiPaths, (0, 1), (0, 1), 0),
      shift(multiPaths, (0, -1), (0, 1), 0),
    ])[:,np.newaxis,:,:], mandatoryAndVisited.shape[0], 1)
    repeatedMandatoryAndVisited = np.repeat(mandatoryAndVisited[np.newaxis,:,:,:], 4, 0)
    shiftedMandatoryMultiPaths = np.all([shiftedMultiPaths, repeatedMandatoryAndVisited], 0)
    multiPathSiblings = np.count_nonzero([
      shift(shiftedMandatoryMultiPaths[0], (-1, 0), (1, 2), 0),
      shift(shiftedMandatoryMultiPaths[1], (1, 0), (1, 2), 0),
      shift(shiftedMandatoryMultiPaths[2], (0, -1), (1, 2), 0),
      shift(shiftedMandatoryMultiPaths[3], (0, 1), (1, 2), 0),
    ], axis=0)

    mandatorySatisfiedMask = ~np.any(multiPathSiblings > 2, axis=(1, 2))

    return forbiddenSatisfied[mandatorySatisfiedMask]
  
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

    bigArrays = np.array([np.concatenate([np.concatenate([x, np.zeros((squareShape[0] - x.shape[0], x.shape[1]))], 0), np.zeros((squareShape[0], squareShape[1] - x.shape[1]))], 1) for x in smallArrays], dtype=np.bool)

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
    True,
    [
      [True],
      [True],
      [True],
      [True],
    ]
  ),(
    'c',
    0,
    True,
    [
      [False, True],
      [False, True],
      [True, True],
    ]
  ),(
    'b',
    0,
    True,
    [
      [True, True],
      [False, True],
    ]
  )
]

globalSpikes = np.array([s2c(l) for l in ['z', 'y', 'x']])
globalSquares = np.array([s2c(l) for l in ['l', 'm', 'n']])
globalPoints = np.array([s2c(l) for l in ['t']])