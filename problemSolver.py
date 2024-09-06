from problemTypes.squareColorAndPairSquares.SquareColorAndPairSquaresProblem import ProblemDefinition
from utils import arrayToPrintable, arrayToTuple, fileToArray, arrayToDebug
from collections import deque
import time

array = fileToArray("problemTypes/squareColorAndPairSquares/maps/actual5.txt")
problem = ProblemDefinition(array)
stateQueue = deque(problem.getStarting())
satisfiedState = None
steps = 0
startTime = time.time()

while len(stateQueue) > 0:
  nextStateAttempt = stateQueue.popleft()
  possibleNextStates = problem.getNexts(nextStateAttempt)
  for possibleNextState in possibleNextStates:
    if problem.isSatisfied(possibleNextState):
      satisfiedState = possibleNextState
      break

    if problem.isUnsatisfiable(possibleNextState):
      continue

    stateQueue.appendleft(possibleNextState)

  if satisfiedState is not None:
    break
  steps += 1

if satisfiedState is not None:
  print("Satisfied")
  print(steps, "steps")
  print(f"{time.time() - startTime:.3f} seconds")
  print(arrayToPrintable(array, satisfiedState))
else:
  print("Unsatisfiable")
  print(steps, "steps")
  print(f"{time.time() - startTime:.3f} seconds")