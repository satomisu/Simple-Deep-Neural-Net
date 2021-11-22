# =================================
# Main script for training a simple
# feed forward DNN
# =================================

import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from util import name_and_path_util
from util.datahandler import DataHandler
from util.records import Recorder
from models.simple_dnn import SimpleDNN
from models.trainer import Trainer
from util import save_and_load


def main(cmdl_params):

    tf.compat.v1.disable_eager_execution()

    # ==============================
    # Get parameters from the parser
    # ==============================
    output_path = cmdl_params.output_path
    output_dir_name = cmdl_params.out_dir_name
    exp_id = cmdl_params.id
    run_number = cmdl_params.run
    epoch = cmdl_params.epoch
    num_batches = cmdl_params.minibatch
    learning_rate = cmdl_params.lrn_rate
    layer_keys_list = cmdl_params.layer_key_list
    neurons_list = cmdl_params.neurons
    activations_list = cmdl_params.acti
    biases_list = cmdl_params.bias
    training_data_path = cmdl_params.data_file
    label_norm_const = cmdl_params.norm_const

    # =========================
    # Set paths and directories
    # =========================
    # Ge necessary names and paths
    output_dir_path = name_and_path_util.get_output_directory_path(output_path, output_dir_name)
    experiment_name = name_and_path_util.get_experiment_name(exp_id)
    loss_file_prefix = name_and_path_util.get_loss_file_prefix(experiment_name, run_number)

    # Set up output directory
    name_and_path_util.create_directory_if_it_does_not_exist(output_dir_path)

    # =========================
    # Network layers dictionary
    # =========================
    network_layers_dict = make_layers_dict(layer_keys_list,
                                           neurons_list,
                                           activations_list,
                                           biases_list)

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
                        output_path=output_dir_path,
                        measure_name='MSE')

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

    # ===============
    # Train the model
    # ===============
    trainer.train()

    # ===============
    # Collect results
    # ===============
    # Get loss per epoch list and save this
    loss_per_epoch = trainer.get_eval_loss()
    loss_file_path = recorder.save_data_to_pkl(loss_file_prefix, loss_per_epoch)

    # Get best prediction paths:
    best_epoch_paths = trainer.get_best_weights_biases_save_paths()

    # Then look at training results
    # Plot learning curve
    plot_learning_curve(loss_file_path)
    # Plot prediction vs. truth
    plot_prediction_vs_truth(best_epoch_paths[-1])


# =====
# Plots
# =====
def plot_learning_curve(loss_file_path):
    def get_file_prefix():
        temp_list = loss_file_path.split('.')
        temp_list_len = len(temp_list)
        temp_list = temp_list[0:temp_list_len-1]
        return '.'.join(temp_list)

    # Get output path prefix
    file_prefix = get_file_prefix()

    # Load loss data
    loss_per_epoch = save_and_load.load_pkl_data(loss_file_path)

    # Make x-axis data
    epoch = len(loss_per_epoch)
    x = np.arange(0, epoch)

    # Plot and save
    plt.clf()
    fig = plt.figure()
    plt.plot(x, np.array(loss_per_epoch))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig.savefig(f'{file_prefix}.png', bbox_inches='tight')


def plot_prediction_vs_truth(file_path):
    def get_file_prefix():
        temp_list = file_path.split('.')
        temp_list_len = len(temp_list)
        temp_list = temp_list[0:temp_list_len-1]
        return '.'.join(temp_list)

    # Get output file prefix
    file_prefix = get_file_prefix()

    # Load data
    data_list = save_and_load.load_pkl_data(file_path)
    prediction_data_dict = data_list[-1]
    label = prediction_data_dict['label']
    prediction = prediction_data_dict['prediction']

    # Plot and save
    plt.clf()
    fig = plt.figure()
    plt.plot(prediction, label, marker='.', ls=' ')
    plt.xlabel('truth')
    plt.ylabel('prediction')
    fig.savefig(f'{file_prefix}_prediction_vs_truth.png', bbox_inches='tight')


# ====
# Util
# ====
# This function makes a dictionary with network structure detail for SimpleDNN object.
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
    # get_data_array returns training and eval inputs and labels.
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


# Modify this function for a data file at hand.
# Output should not be modified.
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

    # Make a list of booleans
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
    cmdl_parser.add_argument('output_path',
                             type=str,
                             help='Output path for experiment results. This will be pre-pended to out_dir_name.')

    cmdl_parser.add_argument('out_dir_name',
                             type=str,
                             help='Output directory name. e.g. Test.'
                             )

    cmdl_parser.add_argument('id',
                             type=str,
                             help='Integer indicating experiment ID.'
                             )

    cmdl_parser.add_argument('run',
                             type=int,
                             help='Run (trial) number of each experiment'
                             )

    # ========================
    # Required non-positionals
    # ========================
    cmdl_parser.add_argument('-data_file',
                             type=str,
                             required=True,
                             help='Path to training data file.'
                             )

    cmdl_parser.add_argument('-epoch',
                             type=int,
                             required=True,
                             help='Training epoch.'
                             )

    cmdl_parser.add_argument('-minibatch',
                             type=int,
                             required=True,
                             help='Number of minibatches.'
                             )

    cmdl_parser.add_argument('-lrn_rate',
                             type=float,
                             required=True,
                             help='Learning rate.'
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
                             help='A list for activation functions for each layer. Order: from the first hidden layer to output.'
                             )

    # ========
    # Optional
    # ========
    cmdl_parser.add_argument('-norm_const',
                             type=float,
                             required=False,
                             default=1,
                             help='Normalization constant for label. Default is unity.')

    cmdl_parser.add_argument('-P',
                             action='store_true',
                             default=False,
                             help='Flag this option to see a summary of parsed args.')

    cmdl_parser.add_argument('-bias',
                             action='store_true',
                             default=False,
                             help='Flag when using bias.'
                             )

    # =============
    # Place holders
    # =============
    cmdl_parser.add_argument('-layer_key_list',
                             default=[]
                             )
    # ===============
    # Parse arguments
    # ===============
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

    # Format some args
    args.bias = make_bool_list(len(args.neurons), bool_value=args.bias)
    args.layer_key_list = make_keys_list(len(args.neurons), prefix='HL')
    args.acti = replace_None(args.acti)

    # =====
    # Print
    # =====
    if args.P:
        print('=== Output Data Info ===')
        print(f'{"output dir":>30}: {args.out_dir_name:<}')
        print(f'{"experiment ID #":>30}: {args.id:<01}')
        print(f'{"experiment run #":>30}: {args.run:<01}')

        print('\n')
        print('=== Data ===')
        print(f'{"Data Path":>30}: {args.data_file}')
        print(f'{"Label Normalization Constant":>30}: {args.norm_const}')

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
