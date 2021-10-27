import datetime
import numpy as np


# ============
# Config Class
# ============
class Config:
    def __init__(self,
                 exp_name=None,
                 num_batches=100,
                 epoch=5,
                 learning_rate=0.1,
                 save_n_best=True,
                 max_keep=3,
                 save_steps=5,
                 less_loss_is_better=True,
                 data_path=None,
                 output_dir_path=None,
                 normalize_label=True,
                 record_eval_at_each_epoch=True
                 ):

        # ====
        # Copy
        # ====
        self.exp_name = exp_name

        # Training specs
        self.num_batches = num_batches
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.normalize_label = normalize_label

        # Save model and loss matter
        self.save_n_best = save_n_best
        self.max_keep = max_keep
        self.save_step = save_steps
        self.less_is_better = less_loss_is_better
        self.record_eval_at_each_epoch = record_eval_at_each_epoch
        self.loss_name_dict = {}

        # Data path
        self.data_path = data_path
        self.output_dir_path = output_dir_path
        self.weights_biases_output_path = None

        # ==============
        # To be assigned
        # ==============
        self.layer_keys_list = []
        self.neurons_list = []
        self.activations_list = []
        self.biases_list = []
        self.layers_dict = {}
        self.loss_name_dict = {}
        self.label_normalization_const = None

        # =================================
        # Hard coded parameters
        # ---------------------------------
        self.optimizer_name = 'optimizer'
        self.loss_op_name = 'loss'

        # Network input/output dimensions
        self.num_input_features = 14
        self.input_shape_np = [-1, 14]
        self.label_shape_np = [-1, 1]
        self.input_shape_tf = (None, 14)
        self.label_shape_tf = (None, 1)
        # =================================

        # =================
        # Recorder settings
        # =================
        self.current_time = None
        self.exp_dir_name_by_timestamp = None

        # ==============
        # logger setting
        # ==============
        self.logger_setup_dict = {}

    # =================
    # Initialization op
    # =================
    def initialize_recorder_matter(self):
        # =================
        # Recorder settings
        # =================
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir_name_by_timestamp = f'EXPAT{self.current_time}'

        # ==============
        # logger setting
        # ==============
        self.logger_setup_dict = {
                                 'LOGGING': True,
                                 'LOG_MODE': 'debug',
                                 'CREATE_LOG_FILE': False,
                                 'STREAM_LOG': True,
                                 'LOGGER_NAME': 'logger',
                                 'LOG_FILE_NAME': 'log_file',
                                 'LOG_DIR_PATH': f'{self.output_dir_path}/EXP{self.exp_id}LOG'
                                 }

        # ==============================
        # Save weights and biases to PKL
        # ==============================
        self.weights_biases_output_path = f'{self.output_dir_path}/EXP{self.exp_id}WeightsBiases'

    # =======
    # Setters
    # =======
    def set_label_normalization_cost(self, norm_const):
        self.sv_normalization_const = norm_const

    # =======
    # Setters
    # =======
    def set_net_specs(self, layer_keys_list, neurons_list, activations_list, biases_list):
        self.layer_keys_list = layer_keys_list
        self.neurons_list = neurons_list
        self.activations_list = activations_list
        self.biases_list = biases_list

    # =================
    # Initialization op
    # =================
    def initialize(self):
        self._abort_if_incomplete_net_specs()
        self._init_layers_dict()

    # =======================
    # Init baseline net specs
    # =======================
    def _init_layers_dict(self):
        self.layers_dict = {}
        for i in range(len(self.layer_keys_list)):
            layer_key = self.layer_keys_list[i]
            self.layers_dict[layer_key] = super().get_layer_specs_dict(neuron=self.neurons_list[i],
                                                                       acti=self.activations_list[i],
                                                                       bias=self.biases_list[i],
                                                                       layer_name=layer_key)
        super().make_loss_name_dict()

    def _abort_if_incomplete_net_specs(self):
        error_message_list = []
        if len(self.layer_keys_list) == 0:
            error_message_list.append('layer_leys_list empty!')
        if len(self.neurons_list) == 0:
            error_message_list.append('neurons_list empty!')
        if len(self.activations_list) == 0:
            error_message_list.append('activations_list empty!')
        if len(self.biases_list) == 0:
            error_message_list.append('biases_list empty!!')
        if len(error_message_list) > 0:
            for message in error_message_list:
                print(message)
            print('aborting!')
            exit(1)

    # =======
    # Methods
    # =======
    def get_layer_specs_dict(self, neuron=1, acti='relu', bias=True, layer_name='layer'):
        the_dict = {'neurons': neuron,
                    'activation': acti,
                    'bias': bias,
                    'layer_name': layer_name}
        return the_dict

    def make_loss_name_dict(self):
        for robot in self.robot_list:
            self.loss_name_dict[robot] = self.get_loss_name(robot)

    def get_loss_name(self, robot):
        return f'{robot}_loss'
