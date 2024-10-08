import numpy as np

def fileToArray(fileName: str) -> np.ndarray:
  f = open(fileName, "r")
  lines = f.readlines()
  array = np.zeros((len(lines), len(lines[0].strip('\r\n'))), dtype=np.byte)
  for lineIndex, line in enumerate(lines):
    for charIndex, char in enumerate(line.strip('\r\n')):
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

def pad(array: np.ndarray, s, axis, default):
  for padAmount, axis in zip(s, axis):
    padding = np.full_like(array, default)
    paddingAccess = [slice(None)]*array.ndim
    paddingAccess[axis] = [0]
    padding = padding[tuple(paddingAccess)]
    padding = np.repeat(padding, padAmount, axis)
    array = np.concatenate([array, padding], axis)
  return array

def prepad(array: np.ndarray, s, axis, default):
  for padAmount, axis in zip(s, axis):
    padding = np.full_like(array, default)
    paddingAccess = [slice(None)]*array.ndim
    paddingAccess[axis] = [0]
    padding = padding[tuple(paddingAccess)]
    padding = np.repeat(padding, padAmount, axis)
    array = np.concatenate([padding, array], axis)
  return array

def arrayToTuple(array: np.ndarray) -> tuple:
  return tuple(array.reshape((-1)))