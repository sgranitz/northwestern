import pandas as pd

DictA = {'a':1, 'b':2, 'c':3}
DictB = {'a':10, 'b':20, 'c':30}
DictC = {'a':100, 'b':200, 'c':300}

ListODicts = [DictA, DictB, DictC]
print(pd.DataFrame(ListODicts))

DictA = {
  'a':[1, 2, 3, 4],
  'b':[5, 6, 7, 8],
  'c':[10, 9, 8, 7]
}

DictB = {
  'a':[100, 200, 300, 4000],
  'c':[50, 60, 70, 80],
  'g':[1000, 2000, 3000, 0000]
}

FrameA = pd.DataFrame(DictA)
FrameB = pd.DataFrame(DictB)
print(FrameA.append(FrameB, ignore_index=True))
