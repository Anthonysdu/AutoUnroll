import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from scripts.rawDataProducer import get_loops
import yaml
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

benchmark_dir = "../sv-benchmarks/c/"

def read_file(file_path):
    lines = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                lines.append(line.strip())
    except FileNotFoundError:
        print("No file found")
    except Exception as e:
        print("Reading file error", e)
    return lines

def parse_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    result = []

    for rundef in root.findall('rundefinition'):
        rundef_dict = {
            'name': rundef.attrib.get('name'),
            'tasks': []
        }

        for task in rundef.findall('tasks'):
            task_dict = {
                'name': task.attrib.get('name'),
                'includesfile': None,
                'propertyfile': None
            }

            includes = task.find('includesfile')
            if includes is not None:
                task_dict['includesfile'] = includes.text

            propertyfile = task.find('propertyfile')
            if propertyfile is not None:
                task_dict['propertyfile'] = propertyfile.text

            rundef_dict['tasks'].append(task_dict)

        result.append(rundef_dict)

    return result

def get_ymls(set_files):
    all_yml_files = []
    for path in includes_files:
        directory = os.path.dirname(path)
        yml_files = glob.glob(os.path.join(directory, '**', '*.yml'), recursive=True)
        all_yml_files.extend(yml_files)
        #all_yml_files = sorted(set(all_yml_files)
        return all_yml_files

def extract_from_yml(file_path, property):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print(file_path + "is not a valid task")
        return None

    input_file = data.get('input_files')
    properties = data.get('properties', [])

    benchmark = next(
        (p for p in properties if p.get('property_file') == '../properties/unreach-call.prp'),
        None
    )

    if benchmark:
        return {
            'input_files': os.path.join(os.path.dirname(file_path), str(input_file)),
            'expected_verdict': benchmark.get('expected_verdict')
        }
    else:
        print(file_path + " has no matched property")
        return None


def contain_loops(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            content = re.sub(r'//.*', '', content)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            content = re.sub(r'".*?"|\'.*?\'', '', content)
            
            loops = re.findall(r'\b(for|while|do)\b', content)
            
            if len(loops) == 1:
                return True
            else:
                return False

    except FileNotFoundError:
        return False
    
data = parse_xml('esbmc-kind.xml')
includes_files = [task['includesfile'] for task in data[0]['tasks'] if task['includesfile'] is not None]
all_yml_files = get_ymls(includes_files)

benchmarks = []
for yml in all_yml_files:
    benchmark = extract_from_yml(yml, 1)
    if benchmark is not None:
        benchmarks.append(benchmark)

output_data_list = []

def process_item(item):
    file_path = str(item['input_files'])
    expected = str(item['expected_verdict'])
    if expected == 'False':
        loop_count = len(get_loops(file_path))
        return loop_count, {
            'file': file_path,
            'expected_verdict': expected
        }
    return None


loop_dict = defaultdict(list)

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_item, item) for item in benchmarks]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        result = future.result()
        if result is not None:
            loop_count, data = result
            loop_dict[loop_count].append(data)

for loop_count, data_list in loop_dict.items():
    file_name = f'reach_{loop_count}_false.json'
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)