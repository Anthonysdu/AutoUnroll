import json
import os
import yaml
import subprocess
import re
import ast
from collections import defaultdict
import itertools
import concurrent.futures

"""
File - rawDataProducer.py

This file will sample ESBMC running with all combinations of loop unrolling numbers to a defined max bound.
"""

benchmark_folder = []
# directory = "/home/tong/sv-benchmarks/c/"
directory = "/home/tong.wu/GNN/sv-benchmarks/c/"
max_bound = 20
max_workers = 30
timeout = 60
timeout_for_upper_bounds = 60
tool = "../../esbmc"
options_sequencial = "--no-div-by-zero-check --force-malloc-success --state-hashing --add-symex-value-sets --no-align-check --no-vla-size-check --enable-unreachability-intrinsic --no-pointer-check --no-bounds-check --error-label ERROR --goto-unwind"
options_concurrency = "--context-bound 3 --no-pointer-check --no-bounds-check --no-div-by-zero-check --force-malloc-success --state-hashing --add-symex-value-sets --no-align-check --no-vla-size-check --no-por --enable-unreachability-intrinsic --goto-unwind"
final_dir = "../raw_result"
result_dir = "../results"
programs_json = "../reach_3_false.json"
loops_count = int(re.search(r'\d+', programs_json).group())
error_benchmarks = []

def run_command(command, timeout):
    try:
        cmd = f"bash -c 'time -p timeout {timeout}s {command}'"
        # command = "sleep 10"
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        # print(cmd)
        output, error = process.communicate()
        combined_output = error + output
        if process.returncode == 0:
            return combined_output
        else:
            return f"Command execution failed with error: {combined_output}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_loops(files):
    log = (
        run_command(f"{tool} --show-loops --goto-unwind " + files, timeout=timeout)
        .strip()
        .splitlines()
    )
    print(files)
    loops = {}
    for i in range(len(log)):
        line = log[i].strip()

        if line.startswith("goto-loop Loop"):
            # Extract the loop ID
            loop_id = int(line.split()[2][:-1])
            # Extract file information from the next line
            file_info = log[i + 1].strip()
            # Check if the file name matches the input file name
            if files in file_info:
                # Extract the function name
                location = file_info.split()[-5]
                
                # Store loop ID and function name in the loop_map
                loops[loop_id] = location
    return loops


def get_upper_bound(logfile, loops):
    log = logfile
    if "Unwinding recursion" in log:
        return {}, True
    
    if "Symex completed " not in log:
        print("Cannot determin the bounds since symex did not finish.")
        return {}, False

    pattern = r"loop (\d+).*?iteration (\d+).*?line (\d+)"

    unwinds = defaultdict(int)
    matches = re.findall(pattern, log)
    for loop_id, iteration,line in matches:
        line = int(line)
        iteration = int(iteration)
        unwinds[line] = max(unwinds[line], iteration)


    upper_bounds = defaultdict(int)
    for key, val in loops.items():
        if int(val) in unwinds:
            upper_bounds[key] = unwinds[int(val)]

    return upper_bounds, False


def convert_time_to_seconds(time_str):
    # Regular expression to match the time format
    match = re.match(r"(\d+)m([0-9.]+)s", time_str)

    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))

        # Convert total time to seconds
        total_seconds = (minutes * 60) + seconds
        return total_seconds
    else:
        return None


def check_verification_result(output):
    # Default values
    verification_result = "UNKNOWN RESULT"
    real_time = None

    # Use regular expressions to search for verification result keywords
    if re.search(r"VERIFICATION FAILED", output):
        if re.search(r"unwinding assertion loop", output):
            verification_result = "UNWINDING ASSERTION"
        else:
            verification_result = "VERIFICATION FAILED"
    elif re.search(r"VERIFICATION SUCCESSFUL", output):
        verification_result = "VERIFICATION SUCCESSFUL"

    # Extract real time from the output (example: 'real 0.22')
    time_match = re.search(r"real\s+([\d.]+)", output)

    if time_match:
        real_time = float(time_match.group(1))
    return verification_result, real_time

# Run esbmc with unified bound for all loops to find the bound
def get_min_bound_for_bug(options, program):
    command = f"{tool} {options} {program} --32 --no-unwinding-assertions "
    for i in range(1, max_bound):
        cur_command = command + f"--unwind {i}"
        out = run_command(cur_command, timeout=timeout)
        result, time = check_verification_result(out)
        if result == "VERIFICATION FAILED":
            print(f"bounds found bound is {i}")
            return i
    return max_bound

def run_verification_checks(
    program, options=None, max_bound=20, file=None, verdict=None
):
    """
    Run verification checks for a given program using specified unwind bounds.

    Args:
    - program (str): The program file name.
    - max_bound (int): The maximum bound value for the loops. Default is 20.

    Returns:
    - None
    """
    # Retrieve loop information from the program
    program = '../' + program
    loops = get_loops(program)

    if(loops_count != len(loops)):
        print(f"{program}: Actual loops count doesn't match with expected, skip...")
        error_benchmarks.append(program)
        return

    #print(loops)
    # prerun = f"{tool} --unwind {max_bound} --context-bound 2 --goto-unwind {program}"
    prerun = f"{tool} --context-bound 2 --unwind 20 {program}"
    print(prerun)
    upper_bounds, is_recur = get_upper_bound(
        run_command(prerun, timeout=timeout_for_upper_bounds)
    , loops)

    # ESBMC doesn't support recursion for unwindset yet.
    if is_recur and file is not None:
        file.write("doesn't support recursion yet\n")
        file.close()
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        os.rename(file.name, os.path.join(final_dir, os.path.basename(file.name)))
        return
    # upper_bounds = None
    if file is not None:
        file.write(f"upper_bounds {upper_bounds}\n")
        
    # Initialize the array with a value of max_bound for each loop

    bugs_found_bound = get_min_bound_for_bug(options, program)
    print(upper_bounds)
    if upper_bounds:
        bounds = [
            upper_bounds[key] + 1 if key in upper_bounds else max_bound
            for key in loops.keys()
        ]
    else:
        bounds = [max_bound] * len(loops)

    # Generate ranges based on the bounds (from 1 to max_bound for each loop)
    print(bounds)
    ranges = [range(1, bound + 1) for bound in bounds]
    # Generate all combinations of values for the bounds
    all_combinations = itertools.product(*ranges)
    print(loops)
    # Iterate over each combination of values for bounds

    bugs_found = False
    timed_out = False
    for combination in all_combinations:
        unwindset = "--unwindset "
        for (key, value), bound in zip(loops.items(), combination):
            unwindset += f"{key}:{bound},"
        if combination[-1] == 1:
            bugs_found = False
            timed_out = False
        # Remove the trailing comma
        unwindset = unwindset.rstrip(",")

        if timed_out and bugs_found and combination[-1] >= bugs_found_bound:
            file.write(f"{unwindset[12:]} {0} {60}\n")
            continue
        # Construct the command
        if options is None:
            commands = f"{tool} {program} {unwindset} --32"
        else:
            commands = f"{tool} {program} {unwindset} {options} --32"

        commands = commands + " --no-unwinding-assertions"
        # Optionally print the command for debugging
        # print(commands)
        # Run the command and capture the output
        signal = -1
        out = run_command(commands, timeout=timeout)
        result, time = check_verification_result(out)
        if verdict == "True":
            if result == "UNWINDING ASSERTION":
                signal = -1
            elif result == "VERIFICATION SUCCESSFUL":
                signal = 1
            elif result == "VERIFICATION FAILED":
                signal = -2
            else:
                signal = 0
        elif verdict == "False":
            # The loops are not fully unrolled and reports the violation, we need to know
            # whether this unwind factor can really report a bug in the program.
            # if result == "UNWINDING ASSERTION":
            #     signal = -1
            #     commands = commands + " --no-unwinding-assertions"
            #     no_assert_out = run_command(commands, timeout=timeout)
            #     result1, _ = check_verification_result(no_assert_out)
            #     if result1 == "VERIFICATION FAILED":
            #         signal = 1
            if result == "VERIFICATION SUCCESSFUL":
                signal = -1
            # Found a bug in the program.
            elif result == "VERIFICATION FAILED":
                signal = 1
            # Timeout or out of memory
            else:
                signal = 0

        if file is not None and signal != -2:
            file.write(f"{unwindset[12:]} {signal} {time}\n")
            print(file)
            if time > 60:
                timed_out = True
    if file is not None:
        file.close()
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        os.rename(file.name, os.path.join(final_dir, os.path.basename(file.name)))


def run_program_verification(program, verdict, options):
    os.makedirs(result_dir, exist_ok=True)
    program_filename = os.path.basename(program)
    result_file = os.path.join(result_dir, f"{program_filename}_result.txt")
    print(f"Attempting to write to file: {result_file}")
    with open(result_file, "w") as file:
        file.write(f"{program}: {verdict}\n")
        run_verification_checks(
            program,
            options=options,
            max_bound=max_bound,
            file=file,
            verdict=verdict,
        )


if __name__ == "__main__":
    programs = json.load(open(programs_json))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in programs:
            program = item['file']
            expected_verdict = item['expected_verdict']
            future = executor.submit(
                run_program_verification, program, expected_verdict, options_concurrency
            )
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")
    print("error benchmarks: ")
    print(error_benchmarks)