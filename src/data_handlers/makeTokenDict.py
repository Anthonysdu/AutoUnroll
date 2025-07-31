import glob, json, tqdm, time
import numpy as np

files = glob.glob("../../data/graphs/*.json")

tokenSet = set()
breaker = False
for aFile in tqdm.tqdm(files):
    myDict = json.load(open(aFile))
    for key in myDict["tokens"]:
        tokenSet.add(myDict["tokens"][key])

tokenDict = dict()
counter = 0
for item in tokenSet:
    tokenDict[item] = counter
    counter += 1

json.dump(tokenDict, open("../../data/tokenDict.json", "w"))

# graphs = glob.glob("../../data/graphs/*.json")
# tokenDict = json.load(open("../../data/tokenDict.json"))
# graphDict = dict()
# graphDict = json.load(open("../../data/graphs/test.c.json"))
# nodeRepresentations = []
# counter = 0
# tokenToNum = dict()
# for token in graphDict["tokens"]:
#     if token in tokenToNum:
#         continue
#     else:
#         tokenToNum[token] = counter
#         counter+=1
#     aList = np.array([0]*len(tokenDict))
#     aList[tokenDict[graphDict["tokens"][token]]] = 1
#     nodeRepresentations.append(aList)
#     print(nodeRepresentations)
