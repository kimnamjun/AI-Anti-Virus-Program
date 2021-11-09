import json
import lief
import hashlib

file_names = ['C:/Windows/explorer.exe',
              'C:/Windows/py.exe']
json_files = list()

for file_name in file_names:
    js = {'label': -1, 'sha256': [], 'imports': [], 'section': {'entry': '', 'sections': []}}

    with open(file_name, 'rb') as file:
        js['sha256'].append(hashlib.sha256(file.read()).hexdigest())

    binary = lief.parse(file_name)

    # js['section']['entry'] = '.data'
    dic = dict()
    for section in binary.sections:
        dic['name'] = section.name
        dic['entropy'] = section.entropy
        dic['size'] = section.size
        dic['vsize'] = section.virtual_size
        js['section']['sections'].append(dic)

    dic = dict()
    for imp_lib in binary.imports:
        dic[imp_lib.name] = [func.name for func in imp_lib.entries]
        js['section']['sections'].append(dic)

    json_files.append(json.dumps(js))

with open('C:/Users/user/created_features.jsonl', 'w') as file:
    file.writelines(json_files)
