import itertools

loops = {"L1": None, "L2": None, "L3": None}
bounds = [3, 4, 5]
loop_names = list(loops.keys())
ranges = [range(1, b + 1) for b in bounds]
all_combinations = itertools.product(*ranges)
for com in all_combinations:
    if(com[-1] == 1):
        print("reset last iteration from start")
    if(com[-1] == 5):
        print(f"found {com[-1]}")