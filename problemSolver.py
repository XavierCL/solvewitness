import numpy as np
from tqdm import tqdm
from problemTypes.mandatoryLines.mandatoryLineProblem import ProblemDefinition
from utils import arrayToPrintable, arrayToTuple, fileToArray, arrayToDebug
from collections import deque
import time

array = fileToArray("problemTypes/mandatoryLines/maps/actual11.txt")
problem = ProblemDefinition(array)
stateQueue = deque(problem.getStarting())
satisfiedState = None
steps = 0
startTime = time.time()
playedAskedForStop = False

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
    pbar.set_description(f"Queue: {len(stateQueue)}, Path: {[printQueue(i) for i in range(0, 10)]}")

  if steps % 1000 == 0 and steps > 0:
    debugArray = '\n'.join(arrayToDebug(nextStateAttempt))
    print(f"\n{debugArray}")

    nextStateAttempt = stateQueue.popleft()
  else:
    nextStateAttempt = stateQueue.pop()

  possibleNextStates = problem.getNexts(nextStateAttempt)

  for possibleNextState in possibleNextStates:
    if problem.isSatisfied(possibleNextState):
      print("Satisfied")
      print(steps, "steps")
      print(f"{time.time() - startTime:.3f} seconds")
      print(arrayToPrintable(array, possibleNextState))
      satisfiedState = possibleNextState
      con = input("Continue?")

      if con != 'y' and con != 'Y' and con != '1' and con != 'yes':
        playedAskedForStop = True
        break

  nextStates = [s for s in possibleNextStates if not problem.isUnsatisfiable(s)]
  nextStates.reverse()

  for nextState in nextStates:
    stateQueue.append(nextState)

  if playedAskedForStop:
    break
  steps += 1

print("Done exploring tree")

if satisfiedState is None:
  print("Unsatisfiable")
  print(steps, "steps")
  print(f"{time.time() - startTime:.3f} seconds")