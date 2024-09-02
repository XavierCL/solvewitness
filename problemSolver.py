from ConstraintLessProblem import ProblemDefinition
from utils import arrayToPrintable, arrayToTuple, fileToArray, arrayToDebug
from collections import deque

array = fileToArray("maps/constraintLess/simple1.txt")
problem = ProblemDefinition(array)
stateQueue = deque(problem.getStarting())
visited = set()
satisfiedState = None

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

    stateQueue.appendleft(possibleNextState)

  if satisfiedState is not None:
    break

if satisfiedState is not None:
  print("Satisfied")
  print(arrayToPrintable(satisfiedState))
else:
  print("Unsatisfiable")