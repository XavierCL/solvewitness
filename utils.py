import numpy as np

def fileToArray(fileName: str) -> np.ndarray:
  f = open(fileName, "r")
  lines = f.readlines()
  array = np.zeros((len(lines), len(lines[0].strip())), dtype=np.byte)
  for lineIndex, line in enumerate(lines):
    for charIndex, char in enumerate(line.strip()):
      array[lineIndex, charIndex] = s2c(char)
  return array

def arrayToPrintable(starting: np.ndarray, solution: np.ndarray) -> str:
  stringBuilder = ""
  for x in range(len(starting)):
    for y in range(len(starting[x])):
      char = solution[x, y] if solution[x, y] != 0 else starting[x, y]
      stringBuilder += c2s(char) if char != 0 else '*'
    stringBuilder += "\n"
  return stringBuilder

def arrayToDebug(array: np.ndarray) -> list[str]:
  stringBuilder = []
  for line in array:
    stringStringer = ""
    for char in line:
      stringStringer += c2s(char) if char != 0 else '*'
    stringBuilder.append(stringStringer)
  return stringBuilder

def s2c(string: str) -> int:
  return string.encode('utf-8')[0]

def c2s(char: int) -> str:
  return chr(char)

def shift(array: np.ndarray, s, axis, default):
  result = np.roll(array, s, axis)

  for shiftAmount, shiftIndex in zip(s, axis):
    if (shiftAmount == 0):
      continue

    defaultOutIndices = [slice(None)]*array.ndim
    if shiftAmount > 0:
      defaultOutIndices[shiftIndex] = slice(None, shiftAmount)
    else:
      defaultOutIndices[shiftIndex] = slice(shiftAmount, None)

    result[tuple(defaultOutIndices)] = default
    
  return result

def arrayToTuple(array: np.ndarray) -> tuple:
  return tuple(array.reshape((-1)))