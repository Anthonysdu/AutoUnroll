import glob
import os, tqdm, json
import subprocess
from subprocess import Popen, PIPE
import multiprocessing as mp

"""
File - dataCollector.py

This file will run the graph builder on every program in the dataset, minus the too big files
"""

# The files that have the following strings in their name are too big, so we don't include them
tooBig = [
    "Problem06",
    "Problem05",
    "Problem04",
    "Problem09",
    "Problem07",
    "Problem08",
    "Problem19",
    "psyco_cev",
]

files = []
# programs = open("../../data/programs.txt")
# for line in programs:
#     if not os.path.isfile("../../data/raw/" + os.path.basename(line.strip()) + ".txt"):
#         if all(x not in line for x in tooBig):
#             files.append(line.strip())

# print(files)
json_dir = '../../json_result'
filenames = glob.glob(os.path.join(json_dir, "*.json"))
files = []
for f in filenames:
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
        files.append(data['program'])
print(files)
def reachableDefs(cfgDict, genKillDict, stmtToNum, backwardsCFGDict, start):
    reachDefs = dict()
    for key in stmtToNum:
        for item in stmtToNum[key]:
            reachDefs[item] = dict()

    previousReachDef = []
    for _ in range(8):
        frontier = [start]
        visited = []
        while frontier:
            curr = frontier.pop()
            if curr not in cfgDict or curr not in backwardsCFGDict:
                continue

            gen_kill = set()
            stmtNums = stmtToNum[curr]
            for num in stmtNums:
                if num in genKillDict:
                    for item in genKillDict[num]:
                        gen_kill.add(item)

            inSet = dict()
            for node in backwardsCFGDict[curr]:
                for num in stmtToNum[node]:
                    for var in reachDefs[num]:
                        if var in inSet:
                            inSet[var].union(reachDefs[num][var])
                        else:
                            inSet[var] = reachDefs[num][var]

            outSet = inSet

            for item in gen_kill:
                newSet = set()
                newSet.add(item[0])
                outSet[item[1]] = newSet

            for item in cfgDict[curr]:
                if item not in visited:
                    frontier.append(item)
            visited.append(curr)

            for num in stmtNums:
                reachDefs[num] = outSet

        if previousReachDef == reachDefs:
            break
        else:
            previousReachDef = reachDefs

    return reachDefs


def defToRef(reachSet, refs):
    defToRefDict = dict()
    for ref in refs:
        reachable = reachSet[ref]
        if reachable:
            for var in refs[ref]:
                # print(var)
                # print(reachable)
                if var[1] in reachable:
                    for item in reachable[var[1]]:
                        if item in defToRefDict:
                            defToRefDict[item].append(var[0])
                        else:
                            defToRefDict[item] = [var[0]]
    return defToRefDict


def parseGraph(filename, rawGraph):
    ptrToToken = dict()
    astDict = dict()
    cfgDict = dict()
    backwardsCFGDict = dict()
    dfgDict = dict()
    genKillDict = dict()
    refDict = dict()
    stmtToNum = dict()
    tokenSet = set()
    loopToken = dict()

    start = None

    holdOnTo = None
    try:
        for line in rawGraph:
            if "(void)" in line:
                line = "".join(line.split("(void)"))
            newline = "".join(line.strip().split(")"))
            newline = "".join(newline.split("("))
            newline = newline.split(",")

            if newline[0] == "AST":
                # ASTPointer : ASTToken
                if newline[1] not in ptrToToken:
                    ptrToToken[newline[1]] = newline[2]
                if newline[3] not in ptrToToken:
                    ptrToToken[newline[3]] = newline[4]
                # ASTPointer1 : [ASTPointer2, ...]
                if newline[1] in astDict:
                    astDict[newline[1]].append(newline[3])
                else:
                    astDict[newline[1]] = [newline[3]]
            elif newline[0] == "CFG":
                # StatementPointer : StatementPointerSuccessor
                if newline[2] in cfgDict:
                    cfgDict[newline[2]].append(newline[4])
                else:
                    cfgDict[newline[2]] = [newline[4]]
                if ptrToToken[newline[4]] == "Function":
                    holdOnTo = newline[2]
                # The purpose of the backwardsCFGDict is to be able to build the inset
                # for reaching definitions. Since function calls will not add to this
                # set and only serve to cause problems, we're ditching them
                elif holdOnTo:
                    if newline[4] in backwardsCFGDict:
                        backwardsCFGDict[newline[4]].append(holdOnTo)
                    else:
                        backwardsCFGDict[newline[4]] = [holdOnTo]
                    holdOnTo = None
                else:
                    if newline[4] in backwardsCFGDict:
                        backwardsCFGDict[newline[4]].append(newline[2])
                    else:
                        backwardsCFGDict[newline[4]] = [newline[2]]
                if newline[2] in stmtToNum:
                    stmtToNum[newline[2]].append(newline[1])
                else:
                    stmtToNum[newline[2]] = [newline[1]]
                if newline[3] == "main":
                    start = newline[4]

            elif newline[0] == "DFG":
                # DataPoint : DataPointAcceptor
                if newline[1] in dfgDict:
                    dfgDict[newline[1]].append(newline[3])
                else:
                    dfgDict[newline[1]] = [newline[3]]
            elif newline[0] == "Gen/Kill":
                # CFGNum : (DeclRefExpr, Var)
                if newline[1] in genKillDict:
                    genKillDict[newline[1]].append((newline[2], newline[3]))
                else:
                    genKillDict[newline[1]] = [(newline[2], newline[3])]
            elif newline[0] == "Ref":
                # CFGNum : (DeclRefExpr, Var)
                if newline[1] in refDict:
                    refDict[newline[1]].append((newline[3], newline[4]))
                else:
                    refDict[newline[1]] = [(newline[3], newline[4])]
            elif newline[0] == "Loop":
                loopToken[newline[1]] = [(newline[3], newline[4])]

    except:
        print(line)
        assert ()

    reachDef = reachableDefs(cfgDict, genKillDict, stmtToNum, backwardsCFGDict, start)
    defsToRefs = defToRef(reachDef, refDict)
    for aDef in defsToRefs:
        if aDef in dfgDict:
            dfgDict[aDef].append(defsToRefs[aDef])
        else:
            dfgDict[aDef] = [defsToRefs[aDef]]

    output = {
        "tokens": ptrToToken,
        "AST": astDict,
        "ICFG": cfgDict,
        "Data": dfgDict,
        "Loop": loopToken,
    }
    json.dump(
        output,
        open(
            "../../data/graphs/"
            + os.path.basename(filename).removesuffix(".txt")
            + ".json",
            "w",
        ),
    )
    return output

# get loops id from esbmc
def get_loops(file):
    cmd = f"esbmc --show-loops --goto-unwind ../../{file}"
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    output, error = process.communicate()
    log = (error + output).strip().splitlines()
    loops = {}
    for i in range(len(log)):
        line = log[i].strip()
        if line.startswith("goto-loop Loop"):
            # Extract the loop ID
            loop_id = int(line.split()[2][:-1])

            # Extract file information from the next line
            file_info = log[i + 1].strip()

            # Check if the file name matches the input file name
            if os.path.basename(file) in file_info:
                # Extract the line number
                line_number = file_info.split()[3]

                # Store loop ID and line number in the loop_map
                loops[loop_id] = int(line_number)
    return loops

def handler(filename):
    graphBuilder = Popen(
        ["graph-builder", "../../" + filename, "--", "-x", "c"], stdout=PIPE, stderr=PIPE
    )
    stdout, stderr = graphBuilder.communicate()

    if stdout:
        # Call graph-builder and collect output
        stdout = stdout.decode("utf-8")
        graph = stdout.split("\n")
        graph = [x for x in graph if x != ""]
        graph = parseGraph(filename, graph)
        loops = get_loops(filename)
        with open("../../data/loops_data/" + os.path.basename(filename) + "_loops.json", "w") as f:
            json.dump(loops, f, indent=2)
        return None
    else:
        return filename


pool = mp.Pool((mp.cpu_count() * 3) // 4)
result_object = [pool.apply_async(handler, args=([aFile])) for aFile in files]
results = [r.get() for r in tqdm.tqdm(result_object)]

pool.close()
