from problemTypes.doubleColour.DoubleColourProblem import ProblemDefinition
from utils import arrayToPrintable, arrayToTuple, fileToArray, arrayToDebug
from collections import deque

array = fileToArray("problemTypes/doubleColour/maps/actual1.txt")
problem = ProblemDefinition(array)
stateQueue = deque(problem.getStarting())
visited = set()
satisfiedState = None
steps = 0

while len(stateQueue) > 0:
  nextStateAttempt = stateQueue.pop()
  possibleNextStates = problem.getNexts(nextStateAttempt)
  for possibleNextState in possibleNextStates:
    possibleTuple = arrayToTuple(possibleNextState)
    if possibleTuple in visited:
      continue

    visited.add(possibleTuple)

    if problem.isSatisfied(possibleNextState):
      satisfiedState = possibleNextState
      break

    stateQueue.append(possibleNextState)

  if satisfiedState is not None:
    break
  steps += 1

if satisfiedState is not None:
  print("Satisfied in", steps, "steps")
  print(arrayToPrintable(array, satisfiedState))
else:
  print("Unsatisfiable in", steps, "steps")