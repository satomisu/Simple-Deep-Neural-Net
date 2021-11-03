import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from util.records import Recorder
from util.datahandler import DataHandler


# # ==================
# # Trainer Base class
# # ==================
# class TrainerBase:
#     def __init__(self,
#                  transfer=False,
#                  baseline=False
#                  ):
#         self.transfer = transfer
#         self.baseline = baseline
#
#         # Assigned by self.assign_training_specs
#         self.learning_rate = None
#         self.epoch = None
#         self.num_batches = None
#         self.optimizer_name = None
#         self.loss_name = None
#         self.task_list = None
#         self.print_step = 100
#         self.time_1 = None
#         self.exp_name_for_progress_message = None
#
#         # Graph components, initialized by self.initialize_trainer_base()
#         self.graph = None
#         self.graph_initialization_op = None
#         self.model = None
#         self.loss_list = []
#         self.global_step = None
#         self.loss_op = None
#         self.optimizer = None
#         self.optimization_op = None
#
#         # Session, initialized by self.init_session()
#         self.sess = None
#
#         # Recorder object and logger, initialized by self.init_records()
#         self.recorder = None
#         self.logger = None
#
#         # Data
#         self.data_handler = None
#         self.data_path = None
#         self.norm_const = None
#         self.input_shape_np = None
#         self.label_shape_np = None
#         self.placeholder_dict = None
#         self.eval_loss_feed_dict = None
#         self.eval_input_features_feed_dict = None
#         self.eval_sample_size_list = None
#         self.eval_labels_list = None
#
#         # Training record
#         self.record_eval_at_each_epoch = False
#         self.eval_loss_list = []
#         self.eval_prediction_list = []
#         self.eval_prediction_dict = {}
#
#     # =================
#     # Assign attributes
#     # =================
#     def assign_training_specs(self,
#                               task_list=[],
#                               learning_rate=0.1,
#                               epoch=100,
#                               num_batches=100,
#                               optimizer_name='optimizer',
#                               loss_name='loss',
#                               print_step=100,
#                               record_eval_at_each_epoch=True
#                               ):
#         self.task_list = task_list
#         self.learning_rate = learning_rate
#         self.epoch = epoch
#         self.num_batches = num_batches
#         self.optimizer_name = optimizer_name
#         self.loss_name = loss_name
#         self.print_step = print_step
#         self.record_eval_at_each_epoch = record_eval_at_each_epoch
#
#     def assign_model(self, model):
#         self.model = model
#
#     def assign_graph(self, graph):
#         self.graph = graph
#
#     def assign_data_specs(self,
#                           data_path='',
#                           norm_const=1,
#                           input_shape_np=[-1, 14],
#                           label_shape_np=[-1, 1]):
#         self.data_path = data_path
#         self.norm_const = norm_const
#         self.input_shape_np = input_shape_np
#         self.label_shape_np = label_shape_np
#         self.placeholder_dict = {'input_ph_dict': self.model.input_placeholder_dict,
#                                  'label_ph_dict': self.model.label_placeholder_dict
#                                  }
#
#     # ==================
#     # Loss and optimizer
#     # ==================
#     def _build_loss_op(self):
#         label_ph = self.model.label_placeholder_dict[self.task_list[0]]
#         self.loss_op = tf.compat.v1.losses.mean_squared_error(label_ph, self.model.the_model)
#
#     def _build_optimizer(self):
#         self.optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate, name=self.optimizer_name)
#
#     def _build_optimization_op(self):
#         self.global_step = tf.compat.v1.Variable(0, name='global_step')
#         self.optimization_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step)
#
#     # ==================
#     # Initialization ops
#     # ==================
#     def initialize_trainer_base(self):
#         with self.graph.as_default():
#             self._build_loss_op()
#             self._build_optimizer()
#             self._build_optimization_op()
#             self.graph_initialization_op = tf.compat.v1.global_variables_initializer()
#
#         # For naming the experiment for progress report during training
#         joining_str = ' '
#         robot_name_string = joining_str.join(self.task_list)
#         if self.transfer:
#             name = 'transfer to'
#         if self.baseline:
#             name = 'baseline'
#         self.exp_name_for_progress_message = f'{name} {robot_name_string}'
#
#     def init_session(self):
#         self.sess = tf.compat.v1.Session(graph=self.graph)
#         self.sess.run(self.graph_initialization_op)
#
#     def init_records(self, config):
#         self.recorder = RecordsTf1(config,
#                                    self.graph,
#                                    self.sess,
#                                    multitask=False,
#                                    transfer=self.transfer,
#                                    baseline=self.baseline)
#         self.logger = self.recorder.get_logger()
#
#     def init_data(self):
#         self.data_handler = NPZDataHandler(self.data_path,
#                                            task_list=self.task_list,
#                                            placeholder_dict=self.placeholder_dict,
#                                            num_batches=self.num_batches,
#                                            norm_const=self.norm_const,
#                                            multimodal=False,
#                                            multitask=False,
#                                            transfer=self.transfer,
#                                            input_shape_np=self.config.input_shape_np,
#                                            label_shape_np=self.config.label_shape_np
#                                            )
#         self.eval_loss_feed_dict = self.data_handler.get_eval_loss_feed_dict()
#         self.eval_sample_size_list = self.data_handler.get_eval_sample_size_list()
#         self.eval_labels_list = self.data_handler.get_eval_labels_list()
#         self.eval_input_features_feed_dict = self.data_handler.get_eval_input_features_feed_dict()
#
#     def init_data_collection_structures(self):
#         for i in range(len(self.task_list)):
#             self.eval_loss_list.append([])
#
#     # =======
#     # Getters
#     # =======
#     def get_eval_prediction_dictionary(self):
#         return self.eval_prediction_dict
#
#     def get_best_weights_biases_save_paths(self):
#         return self.recorder.weights_biases_data_paths[-1]
#
#     def get_eval_loss_and_task_list(self):
#         return self.eval_loss_list, self.task_list
#
#     def get_prediction_and_task_list(self):
#         prediction = self.predict()
#         return prediction, self.task_list
#
#     def get_best_loss_list(self):
#         return self.recorder.get_measure_list()
#
#     def get_best_epoch_list(self):
#         return self.recorder.get_epoch_list()
#
#     def get_eval_label_list(self):
#         return self.eval_labels_list
#
#     # =======
#     # Methods
#     # =======
#     def train(self):
#         self.print_trainer_info()
#
#         # Train the model
#         self.data_handler.print_data_status()
#
#         total_sgd_steps = self.epoch * self.num_batches
#         sgd_step = 0
#         self.time_1 = time.time()
#         for epoch in range(self.epoch):
#             self.data_handler.batch_training_data()
#             # Train
#             for batch in range(self.num_batches):
#                 # sgd
#                 feed_dict = self.data_handler.get_training_feed_dict(batch)
#                 _, cost = self.sess.run([self.optimization_op, self.loss_op], feed_dict)
#
#                 print_message = self._print_message(sgd_step, self.print_step)
#                 if print_message:
#                     eval_loss = self._get_eval_loss()
#                     self.print_loss_info(sgd_step, total_sgd_steps, 'train step', cost, eval_loss)
#                 sgd_step += 1
#
#             # record evaluation data.
#             current_eval_loss = self._record_eval_data()
#
#             self.save_model(epoch, current_eval_loss)
#
#             self._record_eval_prediction_at_each_epoch()
#             if epoch == (self.epoch - 1):
#                 self._make_eval_prediction_dict_with_label_iunfo()
#
#     def predict(self, return_dict=False):
#         if return_dict:
#             prediction_label_dict = {}
#
#             prediction = self.sess.run(self.model.the_model,
#                                        self.eval_input_features_feed_dict[self.task_list[0]])
#             label = self.eval_labels_list[0]
#             label = np.reshape(label, newshape=prediction.shape)
#             input_dict = self.eval_input_features_feed_dict[self.task_list[0]]
#             input = list(input_dict.values())[0]
#             prediction_label_dict[self.task_list[0]] = {'input': input,
#                                                         'label': label,
#                                                         'prediction': prediction}
#
#             return prediction_label_dict
#
#         else:
#             prediction_list = []
#             prediction_list.append(
#                 self.sess.run(self.model.the_model, self.eval_input_features_feed_dict[self.task_list[0]]))
#
#             return prediction_list
#
#     def save_model(self, epoch, eval_loss):
#         if self.recorder.save_nbest:
#             if self.recorder.nbest_to_be_saved(eval_loss):
#                 weights, biases = self._weights_and_biases_tensors_to_nparray()
#                 prediction_label_dict = self.predict(return_dict=True)
#                 self.recorder.save_nbest_model(eval_loss, epoch, weights, biases, prediction_label_dict)
#         else:
#             if self.recorder.every_savestep_tobe_saved(epoch):
#                 weights, biases = self._weights_and_biases_tensors_to_nparray()
#                 self.recorder.save_model(epoch, weights, biases)
#
#     def initialize_global_variables(self):
#         self.sess.run(self.graph_initialization_op)
#
#     def close_session(self):
#         self.sess.close()
#
#     # ================
#     # Helper Functions
#     # ================
#     def _record_eval_prediction_at_each_epoch(self):
#         if self.record_eval_at_each_epoch:
#             prediction = self.predict(return_dict=False)
#             self.eval_prediction_list.append(prediction[0])
#
#     def _make_eval_prediction_dict_with_label_iunfo(self):
#         if self.record_eval_at_each_epoch:
#             prediction_dict = self.predict(return_dict=True)
#             robot = list(prediction_dict.keys())[0]
#             data_dict = prediction_dict[robot]
#             self.eval_prediction_dict['input'] = data_dict['input']
#             self.eval_prediction_dict['label'] = data_dict['label']
#             self.eval_prediction_dict['predictions'] = self.eval_prediction_list
#
#     def _weights_and_biases_tensors_to_nparray(self):
#         weights_array_dict = {}
#         biases_array_dict = {}
#         for key, tensor in self.model.weights_dict.items():
#             weights_array_dict[key] = self._convert_tensor_to_np(tensor)
#
#         for key, tensor in self.model.biases_dict.items():
#             biases_array_dict[key] = self._convert_tensor_to_np(tensor)
#
#         return weights_array_dict, biases_array_dict
#
#     # Converts tensor to np array
#     def _convert_tensor_to_np(self, tensor):
#         return tensor.eval(session=self.sess)
#
#     def _print_message(self, step, print_step):
#         if step % print_step == 0:
#             return True
#         else:
#             return False
#
#     def _record_eval_data(self):
#         loss = self.sess.run(self.loss_op, self.eval_loss_feed_dict[self.task_list[0]])
#         self.eval_loss_list[0].append(loss)
#         return loss
#
#     def _get_eval_loss(self):
#         loss = self.sess.run(self.loss_op, self.eval_loss_feed_dict[self.task_list[0]])
#
#         return loss
#
#     # =======
#     # Utility
#     # =======
#     def print_loss_info(self, step, total_steps, step_name, train_loss, eval_loss):
#         print('\n')
#         print(f'{self.exp_name_for_progress_message}')
#         print(f'{step_name}: {step}/{total_steps}')
#         if step == 0:
#             print(f'time since start: {time.time() - self.time_1:0.2f} sec')
#         else:
#             print(f'time since last message: {time.time() - self.time_1:0.2f}s')
#         self.time_1 = time.time()
#         print(f'train loss: {train_loss:0.5f}')
#         print(f'eval loss:  {eval_loss:0.5f}')
#
#     def print_weights(self):
#         self.model.print_tensor_specs('weights', self.model.weights_dict)
#         self.model.print_tensor_specs('biases', self.model.biases_dict)
#
#     def print_model_trainables(self):
#         self.model.show_model_trainables(self.graph)
#
#     def clear_all_records(self, ckpt=False, tensorboard=False):
#         self.clear_log_dir()
#         self.clear_weights_bias_dir()
#         if ckpt:
#             self.clear_checkpoint_dir()
#         if tensorboard:
#             self.clear_tensorboard_dir()
#
#     def clear_log_dir(self):
#         self.recorder.clear_log_dir()
#
#     def clear_tensorboard_dir(self):
#         self.recorder.clear_tensorboard_dir()
#
#     def clear_checkpoint_dir(self):
#         self.recorder.clear_checkpoint_dir()
#
#     def clear_weights_bias_dir(self):
#         self.recorder.clear_weights_bias_dir()
#
#     def print_trainer_info(self):
#         print('\n')
#         print('=== Trainer Info ===')
#         print(f'task list: {self.task_list}')
#
#     def plot_prediction(self, show=True):
#         prediction_list = self.predict()
#         if len(prediction_list) == len(self.eval_labels_list):
#             if len(self.eval_labels_list) == len(self.task_list):
#                 for i in range(len(prediction_list)):
#                     plt.plot(self.eval_labels_list[i], prediction_list[i], '.', label=self.task_list[i])
#                 for i in range(len(self.task_list)):
#                     plt.plot(self.eval_labels_list[i], self.eval_labels_list[i], '-', label=self.task_list[i])
#                 if show:
#                     plt.legend()
#                     plt.show()
#             else:
#                 print('legends and labels do not match!')
#         else:
#             print('predictions and labels do not match!')
#
#     def plot_eval_loss(self, show=True):
#         if len(self.eval_loss_list) == len(self.task_list):
#             for i in range(len(self.eval_loss_list)):
#                 plt.plot(self.eval_loss_list[i], '.', label=self.task_list[i])
#             plt.legend()
#             if show:
#                 plt.show()
#         else:
#             print('labels and losses do not match!')
#
#


# =====================
# BaselineTrainer Class
# =====================
class Trainer:
    def __init__(self,
                 model,
                 recorder,
                 data_handler,
                 epoch=100,
                 learning_rate=0.1,
                 num_batches=100,
                 optimizer_name='optimizer',
                 loss_name='L2',
                 print_step=100
                 ):

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.num_batches = num_batches
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.print_step = print_step
        self.recorder = recorder
        self.data_handler = data_handler
        self.time_1 = None

        # Graph components, initialized by self.initialize_trainer_base()
        self.graph = None
        self.graph_initialization_op = None
        self.model = None
        self.placeholder_dict = None
        self.loss_list = []
        self.global_step = None
        self.loss_op = None
        self.optimizer = None
        self.optimization_op = None

        # Session, initialized by self.init_session()
        self.sess = None
        self.placeholder_dict = None
        self.eval_loss_feed_dict = None
        self.eval_input_features_feed_dict = None
        self.eval_sample_size_list = None
        self.eval_labels_list = None

        # Training record
        self.eval_loss_list = []
        self.eval_prediction_list = []
        self.eval_prediction_dict = {}

        self._initialization_op(model)

        exit(0)

    # =================
    # Initializaion ops
    # =================
    def _initialization_op(self, model):
        self._init_model_graph(model)
        self.initialize_trainer_base()
        self.init_session()
        self.model.show_model_trainables(self.graph)
        exit(0)
        self.init_eval_data_dict()

    # Tested
    def _init_model_graph(self, model):
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            # get model here
            model.build_model_graph()

        # Assign the model
        self.assign_model(model)
        self.assign_graph(graph)

        self.placeholder_dict = {'input_ph_dict': self.model.input_placeholder,
                                 'label_ph_dict': self.model.output_placeholder
                                 }

    def assign_model(self, model):
        self.model = model

    def assign_graph(self, graph):
        self.graph = graph

    # Tested
    def initialize_trainer_base(self):
        with self.graph.as_default():
            self._build_loss_op()
            self._build_optimizer()
            self._build_optimization_op()
            self.graph_initialization_op = tf.compat.v1.global_variables_initializer()

    def _build_loss_op(self):
        label_ph = self.model.output_placeholder
        self.loss_op = tf.compat.v1.losses.mean_squared_error(label_ph, self.model.the_model)

    def _build_optimizer(self):
        self.optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate, name=self.optimizer_name)

    def _build_optimization_op(self):
        self.global_step = tf.compat.v1.Variable(0, name='global_step')
        self.optimization_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step)

    # Tested
    def init_session(self):
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.sess.run(self.graph_initialization_op)

    # Currently Worked On
    def init_eval_data_dict(self):
        self.eval_loss_feed_dict = self.data_handler.get_eval_loss_feed_dict()
        self.eval_sample_size_list = self.data_handler.get_eval_sample_size_list()
        self.eval_labels_list = self.data_handler.get_eval_labels_list()
        self.eval_input_features_feed_dict = self.data_handler.get_eval_input_features_feed_dict()

    # =======
    # Getters
    # =======
    def get_eval_prediction_dictionary(self):
        return self.eval_prediction_dict

    def get_best_weights_biases_save_paths(self):
        return self.recorder.weights_biases_data_paths[-1]

    def get_eval_loss(self):
        return self.eval_loss_list

    def get_prediction(self):
        prediction = self.predict()
        return prediction

    def get_best_loss_list(self):
        return self.recorder.get_measure_list()

    def get_best_epoch_list(self):
        return self.recorder.get_epoch_list()

    def get_eval_label_list(self):
        return self.eval_labels_list

    # =======
    # Methods
    # =======
    def train(self):
        self.print_trainer_info()

        # Train the model
        self.data_handler.print_data_status()

        total_sgd_steps = self.epoch * self.num_batches
        sgd_step = 0
        self.time_1 = time.time()
        for epoch in range(self.epoch):
            self.data_handler.batch_training_data()
            # Train
            for batch in range(self.num_batches):
                # sgd
                feed_dict = self.data_handler.get_training_feed_dict(batch)
                _, cost = self.sess.run([self.optimization_op, self.loss_op], feed_dict)

                print_message = self._print_message(sgd_step, self.print_step)
                if print_message:
                    eval_loss = self._get_eval_loss()
                    self.print_loss_info(sgd_step, total_sgd_steps, 'train step', cost, eval_loss)
                sgd_step += 1

            # record evaluation data.
            current_eval_loss = self._record_eval_data()

            self.save_model(epoch, current_eval_loss)

            self._record_eval_prediction_at_each_epoch()
            if epoch == (self.epoch - 1):
                self._make_eval_prediction_dict_with_label_iunfo()

    def predict(self, return_dict=False):
        if return_dict:
            prediction_label_dict = {}

            prediction = self.sess.run(self.model.the_model,
                                       self.eval_input_features_feed_dict[self.task_list[0]])
            label = self.eval_labels_list[0]
            label = np.reshape(label, newshape=prediction.shape)
            input_dict = self.eval_input_features_feed_dict[self.task_list[0]]
            input = list(input_dict.values())[0]
            prediction_label_dict[self.task_list[0]] = {'input': input,
                                                        'label': label,
                                                        'prediction': prediction}

            return prediction_label_dict

        else:
            prediction_list = []
            prediction_list.append(
                self.sess.run(self.model.the_model, self.eval_input_features_feed_dict[self.task_list[0]]))

            return prediction_list

    def save_model(self, epoch, eval_loss):
        if self.recorder.save_nbest:
            if self.recorder.nbest_to_be_saved(eval_loss):
                weights, biases = self._weights_and_biases_tensors_to_nparray()
                prediction_label_dict = self.predict(return_dict=True)
                self.recorder.save_nbest_model(eval_loss, epoch, weights, biases, prediction_label_dict)
        else:
            if self.recorder.every_savestep_tobe_saved(epoch):
                weights, biases = self._weights_and_biases_tensors_to_nparray()
                self.recorder.save_model(epoch, weights, biases)

    def initialize_global_variables(self):
        self.sess.run(self.graph_initialization_op)

    def close_session(self):
        self.sess.close()

    # ================
    # Helper Functions
    # ================
    def _record_eval_prediction_at_each_epoch(self):
        if self.record_eval_at_each_epoch:
            prediction = self.predict(return_dict=False)
            self.eval_prediction_list.append(prediction[0])

    def _make_eval_prediction_dict_with_label_iunfo(self):
        if self.record_eval_at_each_epoch:
            prediction_dict = self.predict(return_dict=True)
            robot = list(prediction_dict.keys())[0]
            data_dict = prediction_dict[robot]
            self.eval_prediction_dict['input'] = data_dict['input']
            self.eval_prediction_dict['label'] = data_dict['label']
            self.eval_prediction_dict['predictions'] = self.eval_prediction_list

    def _weights_and_biases_tensors_to_nparray(self):
        weights_array_dict = {}
        biases_array_dict = {}
        for key, tensor in self.model.weights_dict.items():
            weights_array_dict[key] = self._convert_tensor_to_np(tensor)

        for key, tensor in self.model.biases_dict.items():
            biases_array_dict[key] = self._convert_tensor_to_np(tensor)

        return weights_array_dict, biases_array_dict

    # Converts tensor to np array
    def _convert_tensor_to_np(self, tensor):
        return tensor.eval(session=self.sess)

    def _print_message(self, step, print_step):
        if step % print_step == 0:
            return True
        else:
            return False

    def _record_eval_data(self):
        loss = self.sess.run(self.loss_op, self.eval_loss_feed_dict[self.task_list[0]])
        self.eval_loss_list[0].append(loss)
        return loss

    def _get_eval_loss(self):
        loss = self.sess.run(self.loss_op, self.eval_loss_feed_dict[self.task_list[0]])

        return loss

    # =======
    # Utility
    # =======
    def print_loss_info(self, step, total_steps, step_name, train_loss, eval_loss):
        print('\n')
        print(f'{self.exp_name_for_progress_message}')
        print(f'{step_name}: {step}/{total_steps}')
        if step == 0:
            print(f'time since start: {time.time() - self.time_1:0.2f} sec')
        else:
            print(f'time since last message: {time.time() - self.time_1:0.2f}s')
        self.time_1 = time.time()
        print(f'train loss: {train_loss:0.5f}')
        print(f'eval loss:  {eval_loss:0.5f}')

    def print_weights(self):
        self.model.print_tensor_specs('weights', self.model.weights_dict)
        self.model.print_tensor_specs('biases', self.model.biases_dict)

    def print_model_trainables(self):
        self.model.show_model_trainables(self.graph)

    def clear_all_records(self, ckpt=False, tensorboard=False):
        self.clear_log_dir()
        self.clear_weights_bias_dir()
        if ckpt:
            self.clear_checkpoint_dir()
        if tensorboard:
            self.clear_tensorboard_dir()

    def clear_log_dir(self):
        self.recorder.clear_log_dir()

    def clear_tensorboard_dir(self):
        self.recorder.clear_tensorboard_dir()

    def clear_checkpoint_dir(self):
        self.recorder.clear_checkpoint_dir()

    def clear_weights_bias_dir(self):
        self.recorder.clear_weights_bias_dir()

    def print_trainer_info(self):
        print('\n')
        print('=== Trainer Info ===')
        print(f'task list: {self.task_list}')

    def plot_prediction(self, show=True):
        prediction_list = self.predict()
        if len(prediction_list) == len(self.eval_labels_list):
            if len(self.eval_labels_list) == len(self.task_list):
                for i in range(len(prediction_list)):
                    plt.plot(self.eval_labels_list[i], prediction_list[i], '.', label=self.task_list[i])
                for i in range(len(self.task_list)):
                    plt.plot(self.eval_labels_list[i], self.eval_labels_list[i], '-', label=self.task_list[i])
                if show:
                    plt.legend()
                    plt.show()
            else:
                print('legends and labels do not match!')
        else:
            print('predictions and labels do not match!')

    def plot_eval_loss(self, show=True):
        if len(self.eval_loss_list) == len(self.task_list):
            for i in range(len(self.eval_loss_list)):
                plt.plot(self.eval_loss_list[i], '.', label=self.task_list[i])
            plt.legend()
            if show:
                plt.show()
        else:
            print('labels and losses do not match!')




