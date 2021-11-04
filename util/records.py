import os
from util.nbest import NBest
from util import name_and_path_util
from util import save_and_load


class Recorder:
    def __init__(self,
                 save_nbest,
                 nbest,
                 max_keep=5,
                 save_step=10,
                 less_is_better=True,
                 exp_name='EXP01R01',
                 exp_path='.',
                 measure_name='MSE'):

        self.MAX_KEEP = max_keep
        self.SAVE_STEP = save_step
        self.LESS_IS_BETTER = less_is_better
        self.measure_name = measure_name

        # nbest
        self.save_nbest = save_nbest
        self.nbest = nbest

        # Output paths
        self.exp_name = exp_name
        self.exp_outpath = exp_path
        self.records_output_path = None
        self.output_file_prefix = None

        # Lists of paths_list:
        self.measure_list = []
        self.epoch_list = []
        self.records_output_paths_list = []

        # Initialization
        self._initialize()
        self.print_recorder_setup()

    # =================
    # initalization ops
    # =================
    def _initialize(self):
        self._init_nbest()
        self._init_records_output_path()
        self._init_output_dir()

    def _init_records_output_path(self):
        self.records_output_path = f'{self.exp_outpath}/{self.exp_name}'
        self.output_file_prefix = f'model_weights_biases_{self.measure_name}'

    def _init_output_dir(self):
        name_and_path_util.create_directory_if_it_does_not_exist(self.records_output_path)

    def _init_nbest(self):
        if self.save_nbest:
            self.nbest = NBest(N=self.MAX_KEEP, less_is_better=self.LESS_IS_BETTER)
        else:
            self.nbest = None

    # =====================================
    # Print the output path and file prefix
    # =====================================
    def print_recorder_setup(self):
        print('\n')
        print('=== Recorder Status ============================')
        print(f'Training records will be found at: {self.records_output_path}')
        print(f'Saved model file prefix: {self.output_file_prefix}')
        print(f'output is a plk file containing a list: [weights_dict, biases_dict, prediction_label_dict]')
        print('================================================')
        print('\n')

    # ============
    # Getters
    # ============
    def get_measure_list(self):
        return self.measure_list

    def get_epoch_list(self):
        return self.epoch_list

    def get_records_output_path_list(self):
        return self.records_output_paths_list

    # ===================
    # n-best to be saved?
    # ===================
    def nbest_to_be_saved(self, new_measure):
        return self.nbest.save_model(self.measure_list, new_measure)

    # ===========================
    # every save-step to be saved
    # ===========================
    def every_savestep_tobe_saved(self, epoch):
        if epoch % self.SAVE_STEP == 0:
            return True
        else:
            return False

    # ================
    # Helper functions
    # ================
    def add_measure(self, value):
        self.measure_list.append(value)

    def add_epoch(self, epoch):
        self.epoch_list.append(epoch)

    def add_weights_bias_path(self, path):
        self.records_output_path.append(path)

    def truncate_lists(self):
        while len(self.measure_list) > self.MAX_KEEP:
            self.measure_list.pop(0)
            self.epoch_list.pop(0)
            delete_weights_biases = self.records_output_path.pop(0)
            self.delete_file(delete_weights_biases)

    def save_model(self, epoch, weights, biases):
        file_prefix = f'{self.output_file_prefix}_E{epoch}'
        self.save_records(file_prefix, weights, biases)
        self.truncate_lists()

    def save_nbest_model(self, new_measure, epoch, weights, biases, prediction_label_dict):
        file_prefix = f'{self.output_file_prefix}_E{epoch}'
        self.add_measure(new_measure)
        self.add_epoch(epoch)
        self.save_records(file_prefix, weights, biases, prediction_label_dict)
        self.measure_list, self.epoch_list, self.records_output_path, delete_list = self.nbest.pop_worst(self.measure_list,
                                                                                                         self.epoch_list,
                                                                                                         self.records_output_path)
        self.measure_list, self.epoch_list, self.records_output_path = self.nbest.sort_measure(self.measure_list,
                                                                                               self.epoch_list,
                                                                                               self.records_output_path)

        for a_file in delete_list:
            self.delete_file(a_file)

    # =========================
    # saving weights and biases
    # =========================
    def delete_file(self, file_path):
        os.system(f'rm {file_path}')

    def save_records(self, file_name_prefix: str, weights: dict, biases: dict, prediction_label_dict: dict):
        # weights and biases are dictionaries.
        # for each key and value pair, convert the tensor to np array
        save_path = f'{self.records_output_path}/{file_name_prefix}.pkl'
        save_and_load.savePKLdata([weights, biases, prediction_label_dict], save_path)
        self.add_weights_bias_path(save_path)

