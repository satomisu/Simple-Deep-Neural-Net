# ==========================================================
# Nov. 2nd 2021
# Move onto train_model.py workflow.
# Once trained successfully, delete the commented functions
# ==========================================================
import numpy as np
import random
import math
from collections import OrderedDict


class DataHandler:
    def __init__(self,
                 training_input,
                 training_label,
                 evaluation_input,
                 evaluation_label,
                 num_batches: int,
                 norm_const=1,
                 input_feature_dim=14,
                 label_dim=1
                 ):

        # Copy
        self.training_input = np.array(training_input)
        self.training_label = np.array(training_label)
        self.evaluation_input = np.array(evaluation_input)
        self.evaluation_label = np.array(evaluation_label)
        self.num_batches = num_batches
        self.norm_const = norm_const
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = label_dim
        self.input_shape_np = [-1, input_feature_dim]
        self.label_shape_np = [-1, label_dim]

        # To be set by initialization ops
        self.num_train_samples = 0
        self.num_eval_samples = 0
        self.batch_size = 0
        self.num_remainder_from_batching = 0
        self.shuffled_training_input = None
        self.shuffled_training_label = None
        self.batched_training_input_list = []
        self.batched_training_label_list = []
        self.training_feed_dict = OrderedDict()

        # Initialize
        self._initialize()

    # =================
    # Initialization op
    # =================
    def _initialize(self):
        self._init_num_samples()
        self._init_batch_size()
        self._init_shuffled_training_data()

    # Assigns number of training and evaluation samples.
    def _init_num_samples(self):
        self.num_train_samples = self.training_input.shape[0]
        self.num_eval_samples = self.evaluation_input.shape[0]

    def _init_eval_data(self):
        self.evaluation_input = self.reshape_nparray(self.evaluation_input,
                                                     shape=[self.num_eval_samples, self.input_feature_dim])
        self.evaluation_label = self.reshape_nparray(self.evaluation_label,
                                                     shape=[self.num_eval_samples, self.output_feature_dim])

    # Computes number of samples in a minibatch.
    def _init_batch_size(self):
        if self.num_train_samples is not 0:
            self.batch_size = math.floor(self.num_train_samples/self.num_batches)
            self.num_remainder_from_batching = self.num_train_samples % self.batch_size
        else:
            print('there is no data.... fatal situation!')
            exit(1)

    def _init_shuffled_training_data(self):
        self.shuffled_training_input = np.zeros(self.num_train_samples).tolist()
        self.shuffled_training_label = np.zeros(self.num_train_samples).tolist()

    def _init_batched_training_data_lists(self):
        self.batched_training_input_list = None
        self.batched_training_label_list = None

    # Print data status
    def print_data_status(self):
        print('\n')
        print('=== Data Status ================================')
        print(f'training sample size: {self.num_train_samples}')
        print(f'eval sample size: {self.num_eval_samples}')
        print(f'batch size: {self.batch_size}')
        print(f'num batches: {self.num_batches}')
        if self.num_remainder_from_batching is 0:
            print('all samples are used :)')
        else:
            print(f'Note: There are {self.num_remainder_from_batching} of unused training samples!')
        print('================================================')

    # ==============
    # Public Methods
    # ==============
    def batch_training_data(self):
        # NOTE: This is written for 2d input data

        # Randomly shuffle data
        self._shuffle_training_data()

        # Initialize batched data list
        self._init_batched_training_data_lists()

        # Batch data
        self.batched_training_input_list = self._batch_data(self.shuffled_training_input)
        self.batched_training_label_list = self._batch_data(self.shuffled_training_label)
        # print(len(self.batched_training_input_list))
        # print(len(self.batched_training_label_list))
        # print(self.batched_training_input_list[0].shape)
        # print(self.batched_training_label_list[0].shape)

    def reshape_nparray(self, np_array, shape=[-1, 1]):
        return np_array.reshape(shape[0], shape[1])

    # ================
    # Helper functions
    # ================
    def _batch_data(self, data_array):
        # Only deals with upto 2-d array!!
        # Get dimensions of the array
        array_dim = len(data_array.shape)

        # To be returned
        batched_data_array_list = []

        # Counter
        begin = 0

        # Batch data
        for batch_num in range(self.num_batches):
            # When 2-d array
            if array_dim == 2:
                data_batch = data_array[begin:begin+self.batch_size, :]
                num_features = len(data_batch[0].tolist())
            # When 1-d
            elif array_dim == 1:
                data_batch = data_array[begin:begin+self.batch_size]
                num_features = 1    # If array dimension is one, then number of features is 1
            else:
                print('This batching function only handles upto 2d data!!')
                exit(1)
            data_batch = self.reshape_nparray(data_batch,
                                              shape=[self.batch_size, num_features])
            batched_data_array_list.append(data_batch)

            begin += self.batch_size

        return batched_data_array_list

    def _shuffle_training_data(self):
        # Randomly shuffle the sample index
        indices = list(range(self.num_train_samples))
        random.shuffle(indices)

        # Initialize shuffled training input and label
        self._init_shuffled_training_data()

        # Shuffle the sample
        for i in range(len(indices)):
            index = indices[i]
            self.shuffled_training_input[i] = self.training_input[index, :]
            self.shuffled_training_label[i] = self.training_label[index]

        # Convert back to array
        self.shuffled_training_input = np.array(self.shuffled_training_input)
        self.shuffled_training_label = np.array(self.shuffled_training_label)

    def get_batched_training_input_list(self):
        return self.batched_training_input_list

    def get_batched_training_label_list(self):
        return self.batched_training_label_list

    # ==================
    # Helper^2 functions
    # ==================
    # def _list_of_list_to_list_of_nparryas(self, list_to_convert=[]):
    #     dummy_list = []
    #     for element in list_to_convert:
    #         dummy_list.append(np.array(element))
    #     del list_to_convert
    #     return dummy_list

    # def _get_features_and_label_copy_as_list(self, shuffled_features=[], shuffled_labels=[]):
    #     features_list = []
    #     for features in self.training_features_list:
    #         features_copy = features.copy()
    #         features_list.append(features_copy.tolist())
    #         shuffled_features.append([])
    #
    #     labels_list = []
    #     for label in self.training_labels_list:
    #         label_copy = label.copy()
    #         labels_list.append(label_copy.tolist())
    #         shuffled_labels.append([])
    #
    #     return features_list, labels_list, shuffled_features, shuffled_labels

    # =======
    # Methods
    # =======
    # def get_training_feed_dict(self, batch_num: int, multitask_grad_balance=False):
    #     self.training_feed_dict = {}
    #     if self.multimodal:
    #         pass
    #     elif self.multitask:
    #         input_feature = self.batched_training_features_list[0][batch_num]
    #         if multitask_grad_balance:
    #             for i in range(len(self.task_list)):
    #                 task = self.task_list[i]
    #                 label = self.batched_training_labels_list[i][batch_num]
    #                 self.training_feed_dict[task] = {self.input_placeholder_dict['input']: self._reshape_nparray(input_feature, shape=self.input_shape_np),
    #                                                  self.label_placeholder_dict[task]: self._reshape_nparray(label, shape=self.label_shape_np)
    #                                                  }
    #         else:
    #             self.training_feed_dict[self.input_placeholder_dict['input']] = self._reshape_nparray(input_feature, shape=self.input_shape_np)
    #             for i in range(len(self.task_list)):
    #                 task = f'{self.task_list[i]}'
    #                 label = self.batched_training_labels_list[i][batch_num]
    #                 self.training_feed_dict[self.label_placeholder_dict[task]] = self._reshape_nparray(label, shape=self.label_shape_np)
    #     else:
    #         input_feature = self.batched_training_features_list[0][batch_num]
    #         self.training_feed_dict[self.input_placeholder_dict['input']] = self._reshape_nparray(input_feature,                                                                                                      shape=self.input_shape_np)
    #
    #         for i in range(len(self.task_list)):
    #             task = f'{self.task_list[i]}'
    #             label = self.batched_training_labels_list[i][batch_num]
    #             self.training_feed_dict[self.label_placeholder_dict[task]] = self._reshape_nparray(label,
    #                                                                                                shape=self.label_shape_np)
    #
    #     return self.training_feed_dict
    #
    # def get_eval_loss_feed_dict(self):
    #     eval_feed_dict = {}
    #     if self.multimodal:
    #         pass
    #     elif self.multitask:
    #         formatted_eval_features = self._reshape_nparray(self.eval_features_list[0], shape=self.input_shape_np)
    #         i = 0
    #         for task in self.task_list:
    #             formatter_eval_label = self._reshape_nparray(self.eval_labels_list[i], shape=self.label_shape_np)
    #             eval_feed_dict[task] = {self.input_placeholder_dict['input']: formatted_eval_features,
    #                                     self.label_placeholder_dict[task]: formatter_eval_label
    #                                     }
    #             i += 1
    #     elif self.transfer:
    #         task = self.task_list[0]
    #         formatted_eval_features = self._reshape_nparray(self.eval_features_list[0], shape=self.input_shape_np)
    #         formatted_eval_label = self._reshape_nparray(self.eval_labels_list[0], shape=self.label_shape_np)
    #         eval_feed_dict[task] = {self.input_placeholder_dict['input']: formatted_eval_features,
    #                                 self.label_placeholder_dict[task]: formatted_eval_label}
    #     else:
    #         task = self.task_list[0]
    #         formatted_eval_features = self._reshape_nparray(self.eval_features_list[0], shape=self.input_shape_np)
    #         formatted_eval_label = self._reshape_nparray(self.eval_labels_list[0], shape=self.label_shape_np)
    #         eval_feed_dict[task] = {self.input_placeholder_dict['input']: formatted_eval_features,
    #                                 self.label_placeholder_dict[task]: formatted_eval_label}
    #
    #     return eval_feed_dict
    #
    # def get_eval_input_features_feed_dict(self):
    #     eval_features_dict = {}
    #     if self.multimodal:
    #         pass
    #     elif self.multitask:
    #         formatted_eval_features = self._reshape_nparray(self.eval_features_list[0], shape=self.input_shape_np)
    #         for task in self.task_list:
    #             eval_features_dict[task] = {self.input_placeholder_dict['input']: formatted_eval_features}
    #     elif self.transfer:
    #         task = self.task_list[0]
    #         formatted_eval_features = self._reshape_nparray(self.eval_features_list[0], shape=self.input_shape_np)
    #         eval_features_dict[task] = {self.input_placeholder_dict['input']: formatted_eval_features}
    #     else:
    #         task = self.task_list[0]
    #         formatted_eval_features = self._reshape_nparray(self.eval_features_list[0], shape=self.input_shape_np)
    #         eval_features_dict[task] = {self.input_placeholder_dict['input']: formatted_eval_features}
    #
    #     return eval_features_dict
    #
    # def get_eval_sample_size_list(self):
    #     sample_size_list = []
    #     for eval_label_samples in self.eval_labels_list:
    #         sample_size_list.append(len(eval_label_samples))
    #     return sample_size_list

