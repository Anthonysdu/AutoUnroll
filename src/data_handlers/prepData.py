import glob, json, tqdm, os, random
import multiprocessing as mp
import numpy as np

"""
File - prepData.py

This file will take the graphs produced by dataFormatter.py and
produce the final representation of the graphs. It will produce
the representation of each node in the graph for the GGNN to
perform calculations on. It will also produce a set of edge files
which will contain the edges
"""

graphs = glob.glob("../../data/graphs/*.json")
# results = json.load(open("../../data/SV-CompResults.json"))
results = [os.path.basename(f) for f in glob.glob("../../json_result/*.json")]
tokenDict = json.load(open("../../data/tokenDict.json"))
#program_with_loops = json.load(open("../../program_with_loops.json"))


# for key in results:
#     print([key.split("|||")[0]])
# exit(0)
def makeFinalRep(graph):
    data_dir = "../../data/final_graphs/" + graph.removesuffix('.json')
    os.makedirs(data_dir, exist_ok=True)
    #print(data_dir)
    graphDict = dict()
    if "../../data/graphs/" + graph in graphs:
        graphDict = json.load(open("../../data/graphs/" + graph))
    else:
        return
    nodeRepresentations = []
    counter = 0
    tokenToNum = dict()
    for token in graphDict["tokens"]:
        # print(token)
        if token in tokenToNum:
            continue
        else:
            tokenToNum[token] = counter
            counter += 1
        aList = np.array([0] * len(tokenDict))
        aList[tokenDict[graphDict["tokens"][token]]] = 1
        nodeRepresentations.append(aList)

    ASTDict = []
    for outNode in graphDict["AST"]:
        for inNode in graphDict["AST"][outNode]:
            assert outNode in tokenToNum
            assert inNode in tokenToNum
            ASTDict.append([tokenToNum[outNode], tokenToNum[inNode]])

    ICFGDict = []
    for outNode in graphDict["ICFG"]:
        for inNode in graphDict["ICFG"][outNode]:
            if outNode not in tokenToNum:
                print(graph)
                print(graphDict["ICFG"][outNode])
                print(outNode)
                assert ()
            assert inNode in tokenToNum
            ICFGDict.append([tokenToNum[outNode], tokenToNum[inNode]])

    DataDict = []
    for outNode in graphDict["Data"]:
        for inNode in graphDict["Data"][outNode]:
            if type(inNode) == list:
                for node in inNode:
                    DataDict.append([tokenToNum[outNode], tokenToNum[node]])
            else:
                if inNode not in tokenToNum:
                    continue
                if outNode not in tokenToNum:
                    continue
                DataDict.append([tokenToNum[outNode], tokenToNum[inNode]])

    # This is the all loops with their line numbers.
    # key: token; value: loop_id
    LoopDict = dict()
    FinalDict = dict()

    for addr, line_info_list in graphDict["Loop"].items():
        if not line_info_list:
            continue
        line_info = line_info_list[0]  # [[" line 741", " max iterations = Unknown"]]
        line_str = line_info[0].strip()
        max_iter_str = line_info[1].strip() if len(line_info) > 1 else "max iterations = Unknown"

        line_number = int(line_str.split()[-1])
        max_iterations = max_iter_str.split("=", 1)[-1].strip()

        addr_stripped = str(addr).strip()
        if addr_stripped in tokenToNum:
            token_id = tokenToNum[addr_stripped]
            LoopDict[token_id] = {
                "line": line_number,
                "max_iterations": max_iterations
            }

    loop_dir = graph.removesuffix('.json') + "_loops.json"
    loopFile = json.load(open("../../data/loops_data/" + loop_dir))
    line_to_loopid = {int(v): int(k) for k, v in loopFile.items()}

    for token, loop_info in LoopDict.items():
        line = loop_info["line"]
        if line in line_to_loopid:
            FinalDict[token] = {
                "loop_id": line_to_loopid[line],
                "max_iterations": loop_info["max_iterations"]
            }

    with open(data_dir + "/" + graph, "w") as json_file:
        json.dump(FinalDict, json_file, indent=4)
    
    # print(LoopDict)
    # print(loopFile)
    # print(FinalDict)
    nodeRepresentations = np.array(nodeRepresentations)
    np.savez_compressed(
        data_dir + "/" + graph + "Edges.npz",
        AST=np.array(ASTDict, dtype="long"),
        ICFG=np.array(ICFGDict, dtype="long"),
        Data=np.array(DataDict, dtype="long"),
    )
    np.savez_compressed(
        data_dir +"/" + graph + ".npz", node_rep=nodeRepresentations
    )

    #node_npz = np.load("../../data/final_graphs/" + graph + ".npz")
    #npz_npz = np.load("../../data/final_graphs/" + graph + "Edges.npz")
    # print(node_npz["node_rep"])
    # print(npz_npz["AST"])


pool = mp.Pool(mp.cpu_count() - 2)
# result_object = [pool.apply_async(makeFinalRep, args=([key.split("|||")[0]])) for key in results]
# result_object = [pool.apply_async(makeFinalRep, args=("test.c.json",))]
result_object = [pool.apply_async(makeFinalRep, args=([key])) for key in results]
thing = [r.get() for r in tqdm.tqdm(result_object)]

pool.close()
