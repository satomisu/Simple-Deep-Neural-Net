import tensorflow as tf


# ===============
# SimpleDNN class
# ===============
class SimpleDNN:
    def __init__(self,
                 layer_keys_list,
                 layer_dict,
                 input_shape_tuple,
                 output_shape_tuple,
                 num_input_features
                 ):

        # ==================
        # Primary Attributes
        # ==================
        # List of keys for each DNN layer. Order: input to output.
        self.layer_keys_list = layer_keys_list
        # layer_dict keys: 'neurons', 'activation', 'bias', 'layer name' = layer_key
        self.layer_dict = layer_dict
        # Number of input features. Assumes input is a vector.
        self.num_input_features = num_input_features
        # Input shape in python tuple, e.g., (100, 14).
        self.input_shape_tuple = input_shape_tuple
        # Output shape in python tuple.
        self.output_shape_tuple = output_shape_tuple

        # =============================
        # Set by self.build_model_graph
        # =============================
        self.the_model = None

        self.input_placeholder = None
        self.output_placeholder = None
        self.single_input_placeholder = None
        self.weights_dict = {}
        self.biases_dict = {}

        self.weights_init_status_list = []

        # ==================
        # Initialization ops
        # ==================
        self.build_model_graph()

    def build_model_graph(self):
        # Initializes input and output placeholders.
        self.init_io_placeholders()
        # Builds a simple dnn.
        self._build_simple_dnn()

    def _build_simple_dnn(self):
        for i in range(len(self.layer_keys_list)):
            layer_key = self.layer_keys_list[i]
            layer_spec = self.layer_dict[layer_key]
            out_size = layer_spec['neurons']
            layer_name = layer_spec['layer_name']
            activation = layer_spec['activation']

            if i == 0:
                # i == 0 is the input and should be ignored.
                pass
            elif i == 1:
                layer = self.build_simple_layer(layer_key=layer_name,
                                                in_neuron=self.num_input_features,
                                                out_neuron=out_size,
                                                activation=activation,
                                                in_feature=self.input_placeholder
                                                )
            else:
                layer = self.build_simple_layer(layer_key=layer_name,
                                                in_neuron=last_layer_out_size,
                                                out_neuron=out_size,
                                                activation=activation,
                                                in_feature=layer
                                                )

            i += 1
            last_layer_out_size = out_size
        self.the_model = layer

    # input_output_placeholders
    def init_io_placeholders(self):
        # Input placeholder
        self.input_placeholder = self.get_placeholder(data_type=tf.float32,
                                                      shape=self.input_shape_tuple,
                                                      name='input')

        # Label placeholder
        self.output_placeholder = self.get_placeholder(data_type=tf.float32,
                                                       shape=self.output_shape_tuple,
                                                       name='output')

    # Builds a simple layer.
    def build_simple_layer(self, layer_key,
                           in_neuron,
                           out_neuron,
                           activation,
                           in_feature,
                           np_weights=None,
                           np_biases=None):

        layer_name = layer_key
        weights_name = f'{layer_key}/weights'
        bias_name = f'{layer_key}/bias'
        shape = (in_neuron, out_neuron)

        weights_tensor = self.get_weights_tensor(np_weights=np_weights,
                                                 variable_weights_trainable=True,
                                                 tensor_shape=shape,
                                                 var_name=weights_name)

        bias_tensor = self.get_bias_tensor(np_bias=np_biases,
                                           variable_bias_trainable=True,
                                           num_biases=out_neuron,
                                           var_name=bias_name)

        if activation == 'relu':
            layer = tf.nn.relu(tf.add(tf.matmul(in_feature, weights_tensor), bias_tensor), name=layer_name)

        elif activation is None:
            layer = tf.add(tf.matmul(in_feature, weights_tensor), bias_tensor, name=layer_name)

        else:
            print(f'{activation} not recognized: implement this!!')
            exit(0)

        self.weights_dict[weights_name] = weights_tensor
        self.biases_dict[bias_name] = bias_tensor

        return layer

    # =====================
    # Basic building blocks
    # =====================
    def get_placeholder(self, data_type=tf.float32, shape=(-1, 1), name='plchd_name'):
        place_holder = tf.compat.v1.placeholder(dtype=data_type, shape=shape, name=name)
        return place_holder

    def get_weights_tensor(self,
                           np_weights=None,
                           np_weights_trainable=True,
                           variable_weights_trainable=True,
                           tensor_shape=(1, 1),
                           var_name='tensor_name'):
        if np_weights is None:
            weights_tensor = tf.compat.v1.get_variable(shape=tensor_shape,
                                                       trainable=variable_weights_trainable,
                                                       initializer=tf.compat.v1.glorot_uniform_initializer,
                                                       name=var_name)
            self.weights_init_status_list.append('weights initialized to random')
        else:
            if np_weights_trainable:
                weights_tensor = tf.compat.v1.Variable(np_weights,
                                                       trainable=np_weights_trainable,
                                                       shape=tensor_shape,
                                                       name=var_name)
            else:
                weights_tensor = tf.compat.v1.constant(np_weights,
                                                       shape=tensor_shape,
                                                       name=var_name)
            self.weights_init_status_list.append('weights initialized to transferred weights')

        return weights_tensor

    def get_bias_tensor(self,
                        np_bias=None,
                        variable_bias_trainable=True,
                        num_biases=1,
                        var_name='tensor_name'):

        if np_bias is None:
            bias_tensor = tf.compat.v1.Variable(tf.zeros(num_biases),
                                                trainable=variable_bias_trainable,
                                                name=var_name)
        else:

            bias_tensor = tf.compat.v1.Variable(np_bias,
                                                trainable=variable_bias_trainable,
                                                name=var_name)

        return bias_tensor

    # ====
    # Util
    # ====
    def show_model_trainables(self, graph):
        with tf.compat.v1.Session(graph=graph):
            # The next two lines return identical list
            # trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            trainables = graph.get_collection_ref(name='trainable_variables')
        print('\n')
        print('=== Trainable Variables ===')
        for element in trainables:
            print(f'name: {element.name}')
            print(f'    is trainable: {element.trainable}')
            print(f'    shape: {element.shape}')

    def show_model_collections(self, graph):
        # Gets names of collection of this graph.
        with tf.compat.v1.Session(graph=graph):
            collection_list = graph.collections
        print('\n')
        print('=== Collections ===')
        for element in collection_list:
            print(element)

    def show_model_variables(self, graph):
        with tf.compat.v1.Session(graph=graph):
            variables = graph.get_collection_ref(name='variables')
        print('\n')
        print('=== Variables ===')
        for element in variables:
            print(element.name)
            print(element.trainable)

    def show_model_ops(self, graph):
        with tf.compat.v1.Session(graph=graph):
            ops = graph.get_collection_ref(name='train_op')
        print('\n')
        print('=== ops ===')
        for element in ops:
            print(element.name)

    def print_tensor_specs(self,
                           tensor_name: str,
                           tensor_dict: dict):
        print('\n')
        print('===========================')
        print(f'=== {tensor_name} ===')
        for weights, value in tensor_dict.items():
            shape = value.shape
            print(f'{weights}: shape: {shape}')
            print(f'{weights}: dtype: {value.dtype}')
        print('===========================')





