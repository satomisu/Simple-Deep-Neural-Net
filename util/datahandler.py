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
        self.random_eval_inputs = None
        self.random_eval_labels = None

        # Initialize
        self._initialize()

    # =================
    # Initialization op
    # =================
    def _initialize(self):
        self._init_num_samples()
        self._init_batch_size()
        self._init_shuffled_training_data()
        self._init_dtype_float32()

    def _init_dtype_float32(self):
        if not self.training_input.dtype == np.float32:
            self.training_input = self.convert_to_float32(self.training_input)
        if not self.training_label.dtype == np.float32:
            self.training_label = self.convert_to_float32(self.training_label)
        if not self.evaluation_input.dtype == np.float32:
            self.evaluation_input = self.convert_to_float32(self.evaluation_input)
        if not self.evaluation_label.dtype == np.float32:
            self.evaluation_label = self.convert_to_float32(self.evaluation_label)

    def convert_to_float32(self, an_array):
        # Only works for upto 2d array
        dim = len(an_array.shape)
        if dim == 1:
            num_elements = len(an_array)
            new_array = np.zeros(num_elements, dtype=np.float32)
            for ele in range(num_elements):
                data = an_array[ele]
                data = np.float32(data)
                new_array[ele] = data
            return new_array

        elif dim == 2:
            num_rows = an_array.shape[0]
            num_cols = an_array.shape[1]
            new_array = np.zeros((num_rows, num_cols), dtype=np.float32)
            for row in range(num_rows):
                for col in range(num_cols):
                    data = an_array[row, col]
                    data = np.float32(data)
                    new_array[row, col] = data
            return new_array
        else:
            print('data dimension is larger than 2!')
            exit(0)

    # Assigns number of training and evaluation samples.
    def _init_num_samples(self):
        self.num_train_samples = self.training_input.shape[0]
        self.num_eval_samples = self.evaluation_input.shape[0]

    def init_eval_data(self):
        rand_eval_indices = random.sample(np.arange(0, self.num_eval_samples).tolist(), self.batch_size)
        randomly_sampled_eval_input = np.zeros((self.batch_size, self.input_feature_dim), dtype=np.float32)
        randomly_sampled_eval_label = np.zeros(self.batch_size, dtype=np.float32)
        i = 0
        for index in rand_eval_indices:
            randomly_sampled_eval_input[i, :] = self.evaluation_input[index, :]
            randomly_sampled_eval_label[i] = self.evaluation_label[index]
            i += 1
        randomly_sampled_eval_label = self.reshape_nparray(randomly_sampled_eval_label,
                                                           shape=[self.batch_size, 1])
        self.random_eval_inputs = randomly_sampled_eval_input
        self.random_eval_labels = randomly_sampled_eval_label

    # Computes number of samples in a minibatch.
    def _init_batch_size(self):
        if self.num_train_samples is not 0:
            self.batch_size = math.floor(self.num_train_samples/self.num_batches)
            self.num_remainder_from_batching = self.num_train_samples % self.batch_size
        else:
            print('there is no data.... fatal situation!')
            exit(1)

        if self.batch_size > self.num_eval_samples:
            print(f'The number of samples in a minibatch needs to be smaller'
                  f'than the number of evaluation samples.'
                  f'This is due to the implementation details.'
                  f'minibatch size: {self.batch_size}'
                  f'eval samples: {self.num_eval_samples}')
            exit(0)

    def _init_shuffled_training_data(self):
        self.shuffled_training_input = np.zeros((self.num_train_samples, self.input_feature_dim))
        self.shuffled_training_label = np.zeros(self.num_train_samples)

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
        indices = list(range(self.num_eval_samples))
        random.shuffle(indices)

        # Initialize shuffled training input and label
        self._init_shuffled_training_data()

        # Shuffle the sample
        for i in range(len(indices)):
            index = indices[i]
            self.shuffled_training_input[i, :] = self.training_input[index, :]
            self.shuffled_training_label[i] = self.training_label[index]

        # Convert back to array
        self.shuffled_training_input = np.array(self.shuffled_training_input)
        self.shuffled_training_label = np.array(self.shuffled_training_label)

    def get_batched_training_input_list(self):
        return self.batched_training_input_list

    def get_batched_training_label_list(self):
        return self.batched_training_label_list
