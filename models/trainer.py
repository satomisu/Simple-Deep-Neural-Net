import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os


# =====================
# BaselineTrainer Class
# =====================
class Trainer:
    def __init__(self,
                 model: object,
                 recorder: object,
                 data_handler: object,
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
        self.eval_label_feed_dict = None
        self.eval_input_features_feed_dict = None
        self.eval_sample_size_list = None
        self.eval_labels_list = None

        # Eval data
        self.eval_data_feed_dict = {}
        self.eval_input_feed_dict = {}

        # Training record
        self.eval_loss_list = []
        self.eval_prediction_list = []
        self.eval_prediction_dict = {}

        self._initialization_op(model)

    # =================
    # Initializaion ops
    # =================
    def _initialization_op(self, model):
        self._init_model_graph(model)
        self.initialize_trainer_base()
        self.init_session()
        self.model.show_model_trainables(self.graph)
        self.model.show_model_collections(self.graph)
        self.model.show_model_variables(self.graph)
        self.model.show_model_ops(self.graph)

    # Tested
    def _init_model_graph(self, model):
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            # get model here
            model.build_model_graph()

        # Assign the model
        self.assign_model(model)
        self.assign_graph(graph)

        self.placeholder_dict = {'input_ph': self.model.input_placeholder,
                                 'output_ph': self.model.output_placeholder
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

    # Tested
    def init_eval_data_dict(self):
        self.eval_data_feed_dict = {self.placeholder_dict['input_ph']: self.data_handler.random_eval_inputs,
                                    self.placeholder_dict['output_ph']: self.data_handler.random_eval_labels}
        self.eval_input_feed_dict = {self.placeholder_dict['input_ph']: self.data_handler.random_eval_inputs}

    # =======
    # Getters
    # =======
    def get_eval_prediction_dictionary(self):
        return self.eval_prediction_dict

    def get_best_weights_biases_save_paths(self):
        return self.recorder.get_records_output_path_list()

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
        # Print trainer info
        self.print_trainer_info()

        # Print the data status
        self.data_handler.print_data_status()

        # Initialize sgd step and time matters
        total_sgd_steps = self.epoch * self.num_batches
        sgd_step = 0
        self.time_1 = time.time()

        # Start training
        for epoch in range(self.epoch):
            self.data_handler.init_eval_data()
            self.init_eval_data_dict()
            self.data_handler.batch_training_data()
            training_inputs = self.data_handler.get_batched_training_input_list()
            training_label = self.data_handler.get_batched_training_label_list()

            # Train
            for batch in range(self.num_batches):
                # sgd
                feed_dict = {self.placeholder_dict['input_ph']: training_inputs[batch],
                             self.placeholder_dict['output_ph']: training_label[batch]}
                _, cost = self.sess.run([self.optimization_op, self.loss_op], feed_dict)

                print_message = self._print_message(sgd_step, self.print_step)

                if print_message:
                    eval_loss = self._get_eval_loss()
                    self.print_loss_info(sgd_step, total_sgd_steps, 'train step', cost, eval_loss)
                sgd_step += 1

            # record evaluation data.
            current_eval_loss = self._record_eval_data()
            self.save_model(epoch, current_eval_loss)

    def predict(self, return_dict=False):
        # Get input feed dict
        _input = self.data_handler.random_eval_inputs
        _label = self.data_handler.random_eval_labels
        feed_dict = {self.model.input_placeholder: _input}
        prediction = self.sess.run(self.model.the_model,
                                   feed_dict)
        if return_dict:

            prediction_dict = {'input': _input,
                               'label': _label,
                               'prediction': prediction}

            return prediction_dict

        else:
            return _input, _label, prediction

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
        loss = self._get_eval_loss()
        self.eval_loss_list.append(loss)
        return loss

    def _get_eval_loss(self):
        feed_dict = {self.model.input_placeholder: self.data_handler.random_eval_inputs,
                     self.model.output_placeholder: self.data_handler.random_eval_labels
                     }

        loss = self.sess.run(self.loss_op, feed_dict)
        return loss

    # =======
    # Utility
    # =======
    def print_loss_info(self, step, total_steps, step_name, train_loss, eval_loss):
        print('\n')
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
        self.clear_weights_bias_dir()

    def clear_weights_bias_dir(self):
        self.recorder.clear_weights_bias_dir()

    def print_trainer_info(self):
        print('\n')
        print('=== Trainer Info ===')

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




