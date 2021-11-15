import os
import json
import lief
import hashlib

def file2pe(file_names: list, save_path: str, verbose=1) -> list:
    """
    파일 이름 목록을 넣으면 수집한 데이터와 비슷한 json 형태로 변환합니다.
    :param file_names: 경로를 포함한 실행 파일 목록입니다.
    :param save_path: save_path 지정 시 생성한 json 파일들을 jsonl 포맷으로 저장합니다.
    :param verbose: {0: 출력 안함, 1: 개수만 출력, 2: 파일 리스트 출력}
    :return: json 파일의 리스트입니다.
    """
    json_files = list()
    success_list = list()
    failure_list = list()

    for file_name in file_names:
        try:
            js = {'label': -1, 'datadirectories': [], 'general': {}, 'header': {'coff': {}, 'optional':{}},
                  'imports': {}, 'section': {'sections': []}}

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

            js['name'] = binary.name

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
            success_list.append(file_name)

        except Exception as err:
            failure_list.append(file_name)

    with open(save_path, 'w') as file:
        for json_file in json_files:
            file.write(json_file)
            file.write('\n')
    print(save_path, '에 jsonl 파일이 저장되었습니다.')

    if verbose == 1:
        print(f'성공 {len(success_list)}개')
        print(f'실패 {len(failure_list)}개')
    elif verbose == 2:
        print(f'성공 {len(success_list)}개', success_list)
        print(f'실패 {len(failure_list)}개', failure_list)

    return json_files
