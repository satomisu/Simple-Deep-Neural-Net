# Simple-Deep-Neural-Net
A simple fully connected, feed-forward deep neural network model for regression. tf1 implementation. <br />
&nbsp;&nbsp;&nbsp;&nbsp; To see input options:&nbsp; python train_model.py -h<br />
&nbsp;&nbsp;&nbsp;&nbsp; See &nbsp; train_dnn_model.sh &nbsp; for a use case example. <br />

**Input**: A 1-d vector<br />
**Output**: Single scalar.<br />

**Network weights** are initialized using tf.compat.v1.glorot_uniform_initializer. To change this, make an appropriate modification in SimpleDNN.get_weights_tensor().<br />

**Loss** is MSE loss as in tf.compat.v1.losses.mean_squared_error(). To use another option, make appropriate changes to Trainer._build_loss_op(). <br />

# Requirement
tensorflow 2 and python 3.7. 

# Sample Data
DATA/Kuka_swept_volume_data.pkl<br />
Load using np.load(file_path).<br /> 
<br /> 
The data dictionary contains:<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; training_input: nparray, (100000, 14)<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; training_label: nparray, (100000, )<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; evaluation_input: nparray, (10000, 14)<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; evaluation_input: nparray, (10000, )<br /> 
<br /> 
**Input**: initial and final robot joint angle configurations for Kuka. 7 joint angles for each configuration.<br />
**Label**: Single scalar swept volume value (in liters) corresponding to the input joint angle pair.   

# Main Scripts
## train_dnn_model.sh
To run your own training, modify each variable in this script accordingly. 

## train_model.py
Called by &nbsp; train_dnn_model.sh. &nbsp; If your training/eval data has a different format than described above, then modify &nbsp; get_data_array() &nbsp; function. This script does not need to be modified otherwise. See this script for use case examples for modules and objects. 

# Packages
## util
### datahandler.py
Implements DataHandler class. See &nbsp; train_model.py &nbsp; for usage example. Shuffles, batches, and normalizes data.

### records.py
Implements Recorder class. See &nbsp; train_model.py &nbsp; for usage example. Keeps track of and records quantities to be recorded, e.g., loss values. 

### nbest.py
Implements NBest class. This is used by Records class. This object was written to keep track of the oder of values stored in a dynamic list. 

### save_and_load.py and name_and_path_util.py
Modules for saving and loading files, path operations, and naming experiments, output names. 

### analyze.py 
Module for analyzing training results.

## models
### simple_dnn.py
Implements SimpleDNN class. This is a fully connected feed forward network. Specify the number of layers, neurons, and activations in train_dnn_model.sh.

### trainer.py
Implements Trainder class. Builds training graph for SimpleDNN and trains the model. 


