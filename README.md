# Simple-Deep-Neural-Net
A simple fully connected, feed-forward deep neural network model for regression. tf1 implementation.<br />
<br />
**Input**: A 1-d vector<br />
**Output**: Single scalar.<br />

**Network weights** are initialized using tf.compat.v1.glorot_uniform_initializer. To change this, make an appropriate modification in SimpleDNN.get_weights_tensor().<br />

**Loss** is MSE loss as in tf.compat.v1.losses.mean_squared_error(). To use another option, make appropriate changes to Trainer._build_loss_op(). <br />

# Sample Data
DATA/Kuka_swept_volume_data.pkl<br />
Load using np.load(file_path).<br /> 
<br /> 
This is a dictionary containing:<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; training_input: nparray, (100000, 14)<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; training_label: nparray, (100000, )<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; evaluation_input: nparray, (10000, 14)<br /> 
&nbsp;&nbsp;&nbsp;&nbsp; evaluation_input: nparray, (10000, )<br /> 
<br /> 
**Input**: initial and final robot joint angle configurations for Kuka. 7 joint angles for each configuration.<br />
**Label**: Single scalar swept volume value (in liters) corresponding to the input joint angle pair.   
