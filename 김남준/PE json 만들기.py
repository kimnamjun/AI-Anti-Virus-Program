import os
import json
import lief
import hashlib

# 나중에 돌릴 때는 try 넣어야 할 수도 있음
def parse_to_json(file_names: list) -> list:
    """
    파일 이름 목록을 넣으면 수집한 데이터와 비슷한 json 형태로 변환합니다.
    :param file_names: 경로를 포함한 실행 파일 목록입니다.
    :return: json 파일의 리스트입니다.
    """
    json_files = list()

    for file_name in file_names:
        js = {'label': -1, 'datadirectories': [], 'general': {}, 'header': {'coff': {}, 'optional':{}}, 'imports': {}, 'section': {'entry': '', 'sections': []}}

        binary = lief.parse(file_name)

        for data_directory in binary.data_directories:
            dic = {'name': data_directory.type.name,
                   'size': data_directory.size,
                   'virtual_address': data_directory.rva}
            js['datadirectories'].append(dic)

        js['exports'] = binary.exported_functions

        js['general']['exports'] = len(binary.exported_functions)
        js['general']['has_debug'] = int(binary.has_debug)
        js['general']['has_relocations'] = int(binary.has_relocations)
        js['general']['has_resources'] = int(binary.has_resources)
        js['general']['has_signature'] = int(binary.has_signatures)
        js['general']['has_tls'] = int(binary.has_tls)
        js['general']['imports'] = len(binary.imports)
        js['general']['size'] = os.path.getsize(file_name)
        js['general']['symbols'] = len(binary.symbols)
        js['general']['vsize'] = binary.virtual_size

        js['header']['coff']['characteristics'] = [x.name for x in binary.header.characteristics_list]
        js['header']['coff']['machine'] = binary.header.machine.name
        js['header']['coff']['timestamp'] = binary.header.time_date_stamps

        js['header']['optional']['dll_characteristics'] = [x.name for x in binary.optional_header.dll_characteristics_lists]
        js['header']['optional']['magic'] = binary.optional_header.magic.name
        js['header']['optional']['major_image_version'] = binary.optional_header.major_image_version
        js['header']['optional']['major_linker_version'] = binary.optional_header.major_linker_version
        js['header']['optional']['major_operating_system_version'] = binary.optional_header.major_operating_system_version
        js['header']['optional']['major_subsystem_version'] = binary.optional_header.major_subsystem_version
        js['header']['optional']['minor_image_version'] = binary.optional_header.minor_image_version
        js['header']['optional']['minor_linker_version'] = binary.optional_header.minor_linker_version
        js['header']['optional']['minor_operating_system_version'] = binary.optional_header.minor_operating_system_version
        js['header']['optional']['minor_subsystem_version'] = binary.optional_header.minor_subsystem_version
        js['header']['optional']['sizeof_code'] = binary.optional_header.sizeof_code
        js['header']['optional']['sizeof_headers'] = binary.optional_header.sizeof_headers
        js['header']['optional']['sizeof_heap_commit'] = binary.optional_header.sizeof_heap_commit
        js['header']['optional']['subsystem'] = binary.optional_header.subsystem.name

        for imp_lib in binary.imports:
            dic = dict()
            dic[imp_lib.name] = [func.name for func in imp_lib.entries]
            js['imports'].update(dic)

        js['section']['entry'] = binary.sections[0].name
        for section in binary.sections:
            dic = {'name': section.name,
                   'entropy': section.entropy,
                   'props': [x.name for x in section.characteristics_lists],
                   'size': section.size,
                   'vsize': section.virtual_size}
            js['section']['sections'].append(dic)

        with open(file_name, 'rb') as file:
            js['sha256'] = hashlib.sha256(file.read()).hexdigest()

        json_files.append(json.dumps(js, sort_keys=True))

    return json_files


def save_to_jsonl(json_files: list, path: str) -> None:
    """
    생성된 json_files를 jsonl 파일로 저장
    :param json_files: parse_to_json으로 생성된 jsonl_files
    :param path: 파일 이름을 포함한 저장 경로
    """
    with open(path, 'w') as file:
        for json_file in json_files:
            file.write(json_file)
            file.write('\n')


# 예시
if __name__ == '__main__':
    file_names = ['C:/Windows/py.exe', 'C:/Windows/explorer.exe', 'C:/Users/user/Desktop/minigame.exe']
    json_files = parse_to_json(file_names)
    save_to_jsonl(json_files, 'C:/Users/user/created_features.jsonl')