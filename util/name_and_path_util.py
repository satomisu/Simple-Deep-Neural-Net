import os


def create_directory_if_it_does_not_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        print(f'Directory created: {dir_path}')
    else:
        print(f'Directory already exists: {dir_path}')


def does_a_file_exist(dir_path, file_name):
    if os.path.isfile(f'{dir_path}/{file_name}'):
        return True
    else:
        return False


