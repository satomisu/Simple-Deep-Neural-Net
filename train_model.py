# =================================
# Main script for training a simple
# feed forward DNN
# =================================

import numpy as np
import tensorflow as tf
import argparse

from util import name_and_path_util
from util.datahandler import DataHandler
from util.records import Recorder
from models.simple_dnn import SimpleDNN
from models.trainer import Trainer


def main(cmdl_params):

    tf.compat.v1.disable_eager_execution()

    # ====================================
    # Get parameters from the parser
    # ====================================
    output_path = cmdl_params.out_dir_path
    output_dir_name = cmdl_params.odir
    exp_id = cmdl_params.id
    run_number = cmdl_params.run
    epoch = cmdl_params.epoch
    num_batches = cmdl_params.minibatch
    learning_rate = cmdl_params.lrn_rate
    layer_keys_list = cmdl_params.layer_key_list
    neurons_list = cmdl_params.neurons
    activations_list = cmdl_params.acti
    biases_list = cmdl_params.bias
    record_eval_pred_at_each_epoch = cmdl_params.save_eval_at_each_epoch
    training_data_path = cmdl_params.dfile
    label_norm_const = cmdl_params.norm_const

    # =========================
    # Set paths and directories
    # =========================
    # Ge necessary names and paths
    output_dir_path = name_and_path_util.get_output_directory_path(output_path, output_dir_name)
    experiment_name = name_and_path_util.get_experiment_name(exp_id)
    best_weights_and_prediction_path_file_prefix = name_and_path_util.get_best_weights_and_prediction_file_prefix(experiment_name, run_number)
    loss_file_prefix = name_and_path_util.get_loss_file_prefix(experiment_name, run_number)
    prediction_at_each_epoch_dict_prefix = name_and_path_util.get_prediction_at_each_epoch_dict_prefix(experiment_name, run_number)

    # Set up output directory
    name_and_path_util.create_directory_if_it_does_not_exist(output_dir_path)

    # =========================
    # Network layers dictionary
    # =========================
    network_layers_dict = make_layers_dict(layer_keys_list,
                                           neurons_list,
                                           activations_list,
                                           biases_list)

    # ===============================
    # To be collected during training
    # ===============================
    best_weights_prediction_paths_list = []

    # =======================================
    # Create DataHandler and Recorder Objects
    # =======================================
    # DataHandler
    data_handler = init_data_handler(training_data_path,
                                     num_batches,
                                     label_norm_const,
                                     input_feature_dim=neurons_list[0],
                                     label_dim=neurons_list[-1]
                                     )

    # Recorder
    recorder = Recorder(save_nbest=True,
                        nbest=3,
                        max_keep=3,
                        save_step=10,
                        less_is_better=True,
                        exp_name=experiment_name,
                        exp_path=output_dir_path,
                        measure_name='L2')

    # =================================
    # Create Model and Trainer Objects
    # =================================

    dnn_model = SimpleDNN(layer_keys_list,
                          network_layers_dict,
                          input_shape_tuple=(data_handler.batch_size, neurons_list[0]),
                          output_shape_tuple=(data_handler.batch_size, neurons_list[-1]),
                          num_input_features=neurons_list[0]
                          )

    trainer = Trainer(dnn_model,
                      recorder,
                      data_handler,
                      epoch=epoch,
                      learning_rate=learning_rate,
                      num_batches=num_batches,
                      optimizer_name='optimizer',
                      loss_name='L2',
                      print_step=100)

    trainer.train()

#     # ============
#     # Run training
#     # ============

#     # Train
#     best_data_path, loss_list, robot_list, eval_pred_dict = train_baseline(config)
#     # Organize data
#     best_weights_prediction_paths_list.append(best_data_path)
#     loss_dict['robot_list'] = robot_list
#     loss_dict['loss_list'] = loss_list
#
#     # save the best training data paths and losses
#     np.save(f'{output_dir_path}/{best_weights_prediction_file_prefix}.npy', best_weights_prediction_paths_list)
#     np.savez(f'{output_dir_path}/{loss_file_prefix}.npz', **{key: val for key, val in loss_dict.items()})
#     np.savez(f'{output_dir_path}/{prediction_at_each_epoch_dict_prefix}.npz', **{key: val for key, val in eval_pred_dict.items()})

# # Train the model
# def train_baseline(config):
#     baseline_model = SimpleDNN(task_list=config.robot_list,
#                                layer_keys_list=config.layer_keys_list,
#                                layer_dict=config.layers_dict,
#                                input_shape_tf=config.input_shape_tf,
#                                label_shape_tf=config.label_shape_tf,
#                                num_input_features=config.num_input_features
#                                )
#     baseline_trainer = BaselineTrainer(config, baseline_model)
#     baseline_trainer.train()
#     baseline_loss, baseline_list = baseline_trainer.get_eval_loss_and_task_list()
#     eval_prediction_at_each_epoch_dict = baseline_trainer.get_eval_prediction_dictionary()
#     baseline_best_loss = baseline_trainer.get_best_loss_list()
#     baseline_best_epoch = baseline_trainer.get_best_epoch_list()
#     baseline_trainer.close_session()
#     baseline_best_data_path = baseline_trainer.get_best_weights_biases_save_paths()
#
#     return baseline_best_data_path, baseline_loss, baseline_list, eval_prediction_at_each_epoch_dict


# ====
# Util
# ====
def make_layers_dict(layer_keys_list,
                     neurons_list,
                     activations_list,
                     biases_list
                     ):
    layers_dict = {}
    for i in range(len(layer_keys_list)):
        layer_key = layer_keys_list[i]
        layers_dict[layer_key] = {'neurons': neurons_list[i],
                                  'activation': activations_list[i],
                                  'bias': biases_list[i],
                                  'layer_name': layer_key
                                  }
    return layers_dict


# ====
# DATA
# ====
def init_data_handler(data_path, num_batches, norm_const, input_feature_dim, label_dim):
    # Get data as nparray
    training_input, training_label, evaluation_input, evaluation_label = get_data_array(data_path)

    # Instantiate DataHandler
    data_handler = DataHandler(training_input,
                               training_label,
                               evaluation_input,
                               evaluation_label,
                               num_batches=num_batches,
                               norm_const=norm_const,
                               input_feature_dim=input_feature_dim,
                               label_dim=label_dim
                               )

    return data_handler


# Modify this function for different data file.
def get_data_array(data_path):
    data_dict = np.load(data_path, allow_pickle=True)
    training_input = data_dict['training_input']
    training_label = data_dict['training_label']
    evaluation_input = data_dict['evaluation_input']
    evaluation_label = data_dict['evaluation_label']
    return training_input, training_label, evaluation_input, evaluation_label


# ======
# PARSER
# ======
def training_parser():
    # Inner functions ===================
    # Replaces string None to python None
    def replace_None(a_list):
        for i in range(len(a_list)):
            if a_list[i] == 'None':
                a_list[i] = None
        return a_list

    # Makes layers keys list
    def make_keys_list(list_len, prefix):
        keys_list = []
        for i in range(list_len):
            keys_list.append(f'{prefix}_{i}')
        keys_list[0] = 'input'
        keys_list[-1] = 'output'
        return keys_list

    # Make a list of boolean
    def make_bool_list(list_len, bool_value=True):
        an_arg = []
        for i in range(list_len):
            an_arg.append(bool_value)
        return an_arg
    # ============================================

    # Instantiate argument parser
    cmdl_parser = argparse.ArgumentParser(prog='train_model.py')

    # ====================
    # Required positionals
    # ====================
    cmdl_parser.add_argument('odir',
                             type=str,
                             help='output directory name'
                             )

    cmdl_parser.add_argument('id',
                             type=str,
                             help='experiment ID'
                             )

    cmdl_parser.add_argument('run',
                             type=int,
                             help='run (trial) number of each experiment'
                             )

    # ========
    # Required
    # ========
    cmdl_parser.add_argument('-dfile',
                             type=str,
                             required=True,
                             help='path to training data file'
                             )

    cmdl_parser.add_argument('-norm_const',
                             type=float,
                             required=False,
                             default=1,
                             help='normalization constant for output. Default is unity')

    cmdl_parser.add_argument('-out_dir_path',
                             required=True,
                             type=str,
                             help='path to data output dir. This will be pre-pended to odir')

    cmdl_parser.add_argument('-epoch',
                             type=int,
                             required=True,
                             help='training epoch'
                             )

    cmdl_parser.add_argument('-minibatch',
                             type=int,
                             required=True,
                             help='number of minibatches'
                             )

    cmdl_parser.add_argument('-lrn_rate',
                             type=float,
                             required=True,
                             help='learning rate'
                             )

    cmdl_parser.add_argument('-neurons',
                             type=int,
                             nargs='*',
                             required=True,
                             help='A list of number of neurons in each network layer, including input and output. Order: from input to output.')

    cmdl_parser.add_argument('-acti',
                             type=str,
                             nargs='*',
                             required=True,
                             help='A list for activation functions for each layer. Order: from input to output.'
                             )

    # ========
    # Optional
    # ========
    cmdl_parser.add_argument('-P',
                             action='store_true',
                             default=False,
                             help='flag this option to see parsed args')

    cmdl_parser.add_argument('-bias',
                             action='store_true',
                             default=False,
                             help='flag when using bias'
                             )

    cmdl_parser.add_argument('-save_eval_at_each_epoch',
                             default=False,
                             action='store_true',
                             help='flag this option when recording eval prediction at each epoch')

    # =============
    # Place holders
    # =============
    cmdl_parser.add_argument('-layer_key_list',
                             default=[]
                             )

    # Parse arguments
    args = cmdl_parser.parse_args()

    # ======================
    # Necessary verification
    # ----------------------
    # Empty list of error messages.
    error_message_list = []

    # 1) Number of neurons and number of activations must match.
    if not len(args.neurons) == len(args.acti):
        error_message = 'number of elements provided to -neurons should be one larger than the number of elements provided to -acti!'
        error_message_list.append(error_message)

    # Add more verification items as needed.

    if len(error_message_list) > 0:
        for message in error_message_list:
            print(message)
        exit(1)
    # ======================

    args.bias = make_bool_list(len(args.neurons), bool_value=args.bias)
    args.layer_key_list = make_keys_list(len(args.neurons), prefix='HL')
    args.acti = replace_None(args.acti)

    # =====
    # Print
    # =====
    if args.P:
        print('=== Output Data Info ===')
        print(f'{"output dir":>30}: {args.odir:<}')
        print(f'{"experiment ID #":>30}: {args.id:<01}')
        print(f'{"experiment run #":>30}: {args.run:<01}')

        print('\n')
        print('=== Data ===')
        print(f'{"Data Path":>30}: {args.dfile}')
        print(f'{"Label Normalization Constant":>30}: {args.dfile}')

        print('\n')
        print('=== Training Info ===')
        print(f'{"epoch":>30}: {args.epoch}')
        print(f'{"number of minibatches":>30}: {args.minibatch}')
        print(f'{"learning rate":>30}: {args.lrn_rate}')

        print('\n')
        print('=== Network Info ===')
        print(f'{"network layer keys":>30}: {args.layer_key_list}')
        print(f'{"network layer neurons":>30}: {args.neurons}')
        print(f'{"network layer activations":>30}: {args.acti}')
        print(f'{"network layer biases":>30}: {args.bias}')

    return args


if __name__ == '__main__':
    exp_params = training_parser()

    main(exp_params)
