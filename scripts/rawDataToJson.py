import json
import os
import glob
import ast


"""
File - rawDataToJson.py

This file converts the raw data to json format.
"""

raw_data_dir = "../raw_result"
json_data_dir = "../json_result"
txt_files = glob.glob(os.path.join(raw_data_dir, "*.txt"))
if __name__ == "__main__":
    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            data = f.read()

        lines = data.strip().splitlines()

        if len(lines) < 4:
            print('file ' + txt_file + ' parse error because it does not have loops or valid content')
            continue
        # file path and verdict
        try:
            file_path, verdict = lines[0].split(": ")
            dict_str = lines[1].split("defaultdict")[1]
            start = dict_str.find("{")
            end = dict_str.rfind("}") + 1
            dict_part = dict_str[start:end]

            # upper bounds from pre-run
            upper_bounds = ast.literal_eval(dict_part)
        except:
            print('Can not extract upper bound from file ' + txt_file)
            upper_bounds = {}
        # unwind factors and labels
        results = []
        for line in lines[2:]:
            part, value_str, score_str = line.split()
            coords = part.split(",")
            entry = {}
            for coord in coords:
                k, v = coord.split(":")
                entry[k] = int(v)
            entry["result"] = [int(value_str), float(score_str)]
            results.append(entry)

        file_name = os.path.basename(file_path)
        if not os.path.isdir(json_data_dir):
            os.makedirs(json_data_dir)

        output_path = os.path.join(json_data_dir, file_name + ".json")

        output = {
            "program": file_path,
            "verdict": verdict,
            "upper_bounds": upper_bounds,
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(output, f)
