import numpy as np
from tqdm import tqdm
from problemTypes.spikesSquareAndGeo.spikesSquareAndGeoProblem import ProblemDefinition
from utils import arrayToPrintable, arrayToTuple, fileToArray, arrayToDebug
from collections import deque
import time

array = fileToArray("problemTypes/spikesSquareAndGeo/maps/test3.txt")
problem = ProblemDefinition(array)
stateQueue = deque(problem.getStarting())
satisfiedState = None
steps = 0
startTime = time.time()

def printQueue(index):
  if len(stateQueue) <= index:
    return ""
  
  return np.count_nonzero(stateQueue[index])

def generator():
  while len(stateQueue) > 0:
    yield

pbar = tqdm(generator())
for _ in pbar:
  if steps % 50 == 0:
    pbar.set_description(f"Queue: {len(stateQueue)}, Path: {[printQueue(i) for i in range(0, 3)]}")

  nextStateAttempt = stateQueue.pop()

  if steps % 1000 == 0:
    print("\n", "\n".join(arrayToDebug(nextStateAttempt)))

  possibleNextStates = problem.getNexts(nextStateAttempt)

  for possibleNextState in possibleNextStates:
    if problem.isSatisfied(possibleNextState):
      satisfiedState = possibleNextState
      break

  nextStates = [s for s in possibleNextStates if not problem.isUnsatisfiable(s)]
  nextStates.reverse()

  for nextState in nextStates:
    stateQueue.append(nextState)

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