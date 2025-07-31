#!/usr/bin/env python3

import copy
import itertools
import json
import random
import typing
from pathlib import Path
import heapq

_INDEPENDENT_FACTORS = False
_CUT_CORNERS = True
_SUBSET_SIZE = 40
_TIMEOUT = 60.0
_RESULTS = Path('json_result/')
_MAX_FACTORS = 4
_GROUPS = 10

class FactorSet(typing.NamedTuple):
    factor1: int = -1
    factor2: int = -1
    factor3: int = -1
    factor4: int = -1


if len(FactorSet._fields) < _MAX_FACTORS:
    raise ValueError('FactorSet size is not big enough')

class ESBMCRes(typing.NamedTuple):
    benchmark: str
    bug: bool
    time: float
    unwind: FactorSet
    max_unwind: FactorSet

random.seed(10)

class Solution:
    def __init__(self, bmarks: set[str]) -> None:
        """Create an empty Solution."""
        self.factors : list[FactorSet] = []
        self.found : set[str] = set()
        self.times = {bmark:0.0 for bmark in bmarks}

    def __lt__(self, other):
        if not isinstance(other, Solution):
            return False
        if len(self.found) < len(other.found):
            return True
        if len(self.found) > len(other.found):
            return False
        return sum(list(self.times.values())) > sum(list(other.times.values()))

    def update(self, selected: list[ESBMCRes], factor: FactorSet):
        """Add the given FactorSet to the solution, plus the additional
        ESBMC results corresponding to that FactorSet.
        """
        self.factors.append(factor)
        for item in selected:
            self.times[item.benchmark] += item.time
            self.times[item.benchmark] = min(self.times[item.benchmark], _TIMEOUT)

        self.found |= {item.benchmark for item in selected if item.bug and self.times[item.benchmark] < _TIMEOUT}

    def filter(self, data: list[ESBMCRes]) -> list[ESBMCRes]:
        """Given a list of possible ESBMC Results, return the results
        for benchmarks whose execution is not complete.
        """
        return [item for item in data if item.benchmark not in self.found and self.times[item.benchmark] < _TIMEOUT]

    def utility(self) -> float:
        """Return a metric for how useful this solution is."""
        return len(self.found) / sum(list(self.times.values()))

    def last_factor(self) -> FactorSet:
        """Return the latest FactorSet we have applied."""
        return self.factors[-1]

    def has(self, factor: FactorSet) -> bool:
        """Return whether we have applied this FactorSet before in this solution."""
        return factor in self.factors

    def copy(self) -> typing.Self:
        """Provide an alias for copy.deepcopy."""
        return copy.deepcopy(self)

    def is_dominated(self, other: typing.Self) -> bool:
        """Return whether this solution is objectively worse that other in every regard."""
        if self.found != other.found:
            return False
        if sum(list(self.times.values())) < 0.98 * sum(list(other.times.values())):
            return False
        for bmark, bmark_time in self.times.items():
            if bmark_time < 0.8 * other.times[bmark]:
                return False
        return True

    def is_dominated_by_any(self, others: list[typing.Self]) -> bool:
        """Return whether this solution is objectively worse by any solution in the others set."""
        return any(self.is_dominated(other) for other in others)


def matches(item: ESBMCRes, factor_set: FactorSet) -> bool:
    """Return whether the given ESBMCRes item has the target unwind factor."""
    for idx in range(_MAX_FACTORS):
        target_value = factor_set[idx]
        if item.max_unwind[idx] != -1:
            target_value = min(target_value, item.max_unwind[idx]+1)
        if item.unwind[idx] not in (target_value, -1):
            return False

    return True

def extract_factors(data: list[ESBMCRes]) -> list[FactorSet]:
    """Get a reasonable set of unwind factor combinations
    given the unwind factors in our dataset.
    """
    possible_values = sorted({item.unwind[idx] for item in data for idx in range(_MAX_FACTORS)})
    possible_values = [value for value in possible_values if value > 0]

    if _INDEPENDENT_FACTORS:
        return [FactorSet(*tpl) for tpl in itertools.product(possible_values, repeat=_MAX_FACTORS)]
    return [FactorSet(*itertools.repeat(value, _MAX_FACTORS)) for value in possible_values]

def sort_choices(data: list[ESBMCRes], unwind_factors: list[FactorSet], solution: Solution) -> list[Solution]:
    lst = []
    for factor in unwind_factors:
        selected = [item for item in data if matches(item, factor)]
        new_solution = solution.copy()
        new_solution.update(selected, factor)
        lst.append(new_solution)

    lst.sort(key=lambda x: x.utility(), reverse=True)
    return lst

def eager_solver(data: list[ESBMCRes]) -> Solution:
    """Return the near-optimal static strategy for unnested loops using an eager search strategy.

    Input: List of ESBMCRes tuples, containing the success and runtime of all ESBMC runs
    Output: A List of unwind factors to use in order to maximise success within the _TIMEOUT
    """
    solution = Solution({item.benchmark for item in data})
    unwind_factors = extract_factors(data)

    if _CUT_CORNERS:
        choices = sort_choices(data, unwind_factors, solution)
        num_selected = _SUBSET_SIZE if len(choices) > (_SUBSET_SIZE * 2)  else len(choices) // 2
        unwind_factors = [choice.last_factor() for choice in choices[:num_selected]]

    while unwind_factors:
        best_solution = sort_choices(data, unwind_factors, solution)[0]
        if len(best_solution.found) == len(solution.found):
            break

        solution = best_solution
        unwind_factors.remove(solution.last_factor())
        data = solution.filter(data)

    return solution

def default_solver(data: list[ESBMCRes]) -> Solution:
    """Return the default ESBMC strategy for setting unwind factor.

    Input: List of ESBMCRes tuples, containing the success and runtime of all ESBMC runs
    Output: A List of unwind factors to use in order to maximise success within the _TIMEOUT
    """
    solution = Solution({item.benchmark for item in data})
    for factor in range(2, 21, 2):
        factor_set = FactorSet(*([factor] * _MAX_FACTORS))
        selected = [item for item in data if matches(item, factor_set)]
        solution.update(selected, factor_set)
        data = solution.filter(data)

    return solution

def exhaustive_solver(data: list[ESBMCRes]) -> Solution:
    """Return the near-optimal static strategy for unnested loops using an eager search strategy.

    Input: List of ESBMCRes tuples, containing the success and runtime of all ESBMC runs
    Output: A Solution containing a list of unwind factors to use in order to maximise success within the _TIMEOUT
    """
    return exhaustive_solver2(data)[0]

def exhaustive_solver2(data: list[ESBMCRes]) -> list[Solution]:
    """Return the top 10 static strategies for unnested loops using an eager search strategy.

    Input: List of ESBMCRes tuples, containing the success and runtime of all ESBMC runs
    Output: A list of Solutions, each containing a list of unwind factors to use in order
    to maximise success within the _TIMEOUT
    """
    solutions : list[Solution] = []
    latest_solutions = [Solution({item.benchmark for item in data})]
    new_solutions : list[Solution] = []
    unwind_factors = extract_factors(data)

    if _CUT_CORNERS:
        choices = sort_choices(data, unwind_factors, latest_solutions[0])
        num_selected = _SUBSET_SIZE if len(choices) > (_SUBSET_SIZE * 2) else len(choices) // 2
        unwind_factors = [choice.last_factor() for choice in choices[:num_selected]]

    factor_to_selected = {factor:[item for item in data if matches(item, factor)] for factor in unwind_factors}

    while latest_solutions:
        for factor,selected in factor_to_selected.items():
            for solution in latest_solutions:
                if solution.has(factor):
                    continue

                new_solution = solution.copy()
                new_items = new_solution.filter(selected)
                new_solution.update(new_items, factor)

                if new_solution.is_dominated(solution):
                    continue

                if new_solution.is_dominated_by_any(new_solutions):
                    continue

                if new_solution.is_dominated_by_any(latest_solutions):
                    continue

                new_solutions.append(new_solution)

        solutions.extend(latest_solutions)
        latest_solutions = new_solutions
        new_solutions = []

    solutions.sort(reverse=True)

    # No solution finds bugs
    if not solutions[0].found:
        raise RuntimeError

    return solutions[:6]

def load_data() -> list[ESBMCRes]:
    data = []
    for fpath in _RESULTS.glob('*.json'):
        with fpath.open(encoding='utf8') as fp:
            data_item = json.load(fp)
            bmark: str = data_item['program']
            upper_bounds: dict[str, int] = data_item['upper_bounds']
            results = data_item['results']
            loopids = [key for key in results[0] if key != 'result']

            if len(loopids) > _MAX_FACTORS:
                raise RuntimeError("More loops than supported")

            while len(loopids) < _MAX_FACTORS:
                loopids.append('None')

            max_factors = FactorSet(*[upper_bounds.get(loopid, -1) for loopid in loopids])
            for result in results:
                unwind_factors = FactorSet(*[result.get(loopid, -1) for loopid in loopids])
                data.append(ESBMCRes(bmark, result['result'][0] == 1, result['result'][1], unwind_factors, max_factors))
    return data


def create_synthetic_data() -> list[ESBMCRes]:
    data = []
    bmarks = [f'bmark{idx}' for idx in range(400)]
    for bmark in bmarks:
        base = random.randint(1, 10)
        scaling = random.random()
        minimum_factor = random.randint(2, 30)
        for factor in range(2, 21):
            data.append(ESBMCRes(bmark, factor >= minimum_factor, pow(base, scaling*factor), FactorSet(factor, -1, -1), FactorSet(20, -1, -1)))
    return data

def print_irregular():
    data = load_data()

    temp = {}
    for item in data:
        if item.unwind[1] != -1:
            continue
        if item.benchmark not in temp:
            temp[item.benchmark] = []
        temp[item.benchmark].append(item)

    changes = []
    for bmark, items in temp.items():
        sorted_items = sorted(items, key = lambda x: x.unwind[0])
        previous_time = 0
        cnt = 0
        for item in sorted_items:
            if item.time < (previous_time - 5):
                cnt += 1
            previous_time = item.time
        changes.append((cnt, bmark))

    changes.sort(reverse=True)
    for cnt, bmark in changes[:25]:
        print(cnt, bmark, [item.time for item in sorted(temp[bmark], key=lambda x: x.unwind[0])])

    return


def find_optimal():
    #data = create_synthetic_data()
    data = load_data()

    all_buggy = {item.benchmark for item in data if item.bug}
    print("Total: ", len(all_buggy))
    with open('found_bugs.total.txt', 'w', encoding='utf8') as fout:
        fout.write(f'Total buggy benchmarks: {len(all_buggy)}\n')
        found = sorted(item.rpartition('/')[2] for item in all_buggy)
        for item in found:
            fout.write(f'{item}\n')
    solution = default_solver(data)
    print("incremental: ", len(solution.found), sum(list(solution.times.values())))
    with open('found_bugs.default.txt', 'w', encoding='utf8') as fout:
        found = sorted(item.rpartition('/')[2] for item in solution.found)
        for item in found:
            fout.write(f'{item}\n')

    solution = exhaustive_solver(data)
    print("exhaustive: ", len(solution.found), sum(list(solution.times.values())))
    print(solution.factors)

    with open('found_bugs.exhaustive.txt', 'w', encoding='utf8') as fout:
        found = sorted(item.rpartition('/')[2] for item in solution.found)
        for item in found:
            fout.write(f'{item}\n')

def find_piecewise_optimals():
    #data = create_synthetic_data()
    data = load_data()

    bmarks = list({item.benchmark for item in data})
    random.shuffle(bmarks)

    best_solutions = []
    for group_idx in range(_GROUPS):
        bmarks_subset = set(bmarks[((group_idx * len(bmarks)) // _GROUPS): ((group_idx + 1) * len(bmarks)) // _GROUPS])
        data_subset = [item for item in data if item.benchmark in bmarks_subset]
        best_solutions.append(exhaustive_solver2(data_subset))

    solutions_cnt = {}
    solutions_found = {}
    for solution_group in best_solutions:
        for solution in solution_group:
            factors = tuple(solution.factors)
            if factors not in solutions_cnt:
                solutions_cnt[factors] = 0
                solutions_found[factors] = 0
            solutions_cnt[factors] += 1
            solutions_found[factors] += len(solution.found)

    for factors, found in solutions_found.items():
        solution = Solution(set(bmarks))
        data_copy = data
        for factor_set in factors:
            selected = [item for item in data_copy if matches(item, factor_set)]
            solution.update(selected, factor_set)
            data_copy = solution.filter(data_copy)

        print(factors, found, solutions_cnt[factors], len(solution.found), sum(list(solution.times.values())))

import random

class GA_Solver:
    def __init__(self, data, population_size=50, generations=100, mutation_rate=0.1):
        self.data = data
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.possible_values = sorted({item.unwind[0] for item in data if item.unwind[0] > 0})
        self.max_factors = _MAX_FACTORS

    def random_individual(self):
        values = [random.choice(self.possible_values) for _ in range(self.max_factors)]
        return FactorSet(*values)

    def fitness(self, individual):
        selected = [item for item in self.data if matches(item, individual)]
        found = sum(1 for item in selected if item.bug and item.time < _TIMEOUT)
        total_time = sum(min(item.time, _TIMEOUT) for item in selected)
        return found - 0.05 * (total_time / _TIMEOUT)

    def crossover(self, parent1, parent2):
        idx = random.randint(1, self.max_factors-1)
        new_values = parent1[:idx] + parent2[idx:]
        return FactorSet(*new_values)

    def mutate(self, individual):
        values = list(individual)
        idx = random.randint(0, self.max_factors-1)
        values[idx] = random.choice(self.possible_values)
        return FactorSet(*values)

    def run(self):
        population = [self.random_individual() for _ in range(self.population_size)]

        for gen in range(self.generations):
            population = sorted(population, key=self.fitness, reverse=True)
            next_gen = population[:self.population_size//5]  # keep top 20%

            while len(next_gen) < self.population_size:
                p1, p2 = random.choices(population[:20], k=2)  # choose the best 2 as parents
                child = self.crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                next_gen.append(child)

            population = next_gen
            print(f"Generation {gen} best fitness: {self.fitness(population[0])}")

        best_individual = max(population, key=self.fitness)
        #print("Best unwind factor found by GA:", best_individual)
        return best_individual

def test_ga_solver():
    data = load_data()
    ga = GA_Solver(data, population_size=200, generations=10, mutation_rate=0.05)
    best_factor = ga.run()

    matched_items = [item for item in data if matches(item, best_factor)]
    found_bugs = [item for item in matched_items if item.bug and item.time < _TIMEOUT]
    total_time = sum(min(item.time, _TIMEOUT) for item in matched_items)

    print(f"GA Best FactorSet: {best_factor} {len(found_bugs)} {total_time}")
    # print(f"Found bugs: {len(found_bugs)}")
    # print(f"Total time: {total_time}")

if __name__ == '__main__':
    find_optimal()
    test_ga_solver()
    #find_piecewise_optimals()
