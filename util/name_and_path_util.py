import os


# Names and file prefixes
def get_experiment_name(exp_id, prefix=None):
    if prefix is None:
        if len(exp_id) == 1:
            return f'EXP0{exp_id}'
        elif len(exp_id) == 2:
            return f'EXP{exp_id}'
        else:
            return f'EXP{exp_id}'
    else:
        return f'{prefix}_EXP{exp_id}'


def get_prediction_at_each_epoch_dict_prefix(exp_name, run_number):
    prediction_file_name = f'{exp_name}_R{run_number}_prediction_per_epoch_dict'
    return prediction_file_name


def get_loss_file_prefix(exp_name, run_number):
    loss_file_prefix = f'{exp_name}_R{run_number}_losses_dict'
    return loss_file_prefix


def get_best_weights_and_prediction_file_prefix(exp_name, run_number):
    best_weights_prediction_file_name = f'{exp_name}_R{run_number}_best_weights_and_prediction_paths_list'
    return best_weights_prediction_file_name


# Paths
def get_output_directory_path(outdir_path, dir_name):
    return f'{outdir_path}/{dir_name}'


# Directories and files
def create_directory_if_it_does_not_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        print('\n')
        print(f'Directory created: {dir_path}')
    else:
        print('\n')
        print(f'Directory already exists: {dir_path}')


def does_a_file_exist(dir_path, file_name):
    if os.path.isfile(f'{dir_path}/{file_name}'):
        return True
    else:
        return False


