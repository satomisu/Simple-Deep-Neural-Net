# #############################
# Train a simple neural network
# #############################

# Organize experiments
out_dir_path='.'  # Output directory path
ODIR='test' # Directory name for output data.
ID=1        # Experiment ID
R=1         # Run number

# Data path
data_path='./DATA/Kuka_swept_volume_data.pkl'
# Normalization constant for the training label
norm_const=311.17

# Network parameters
neur="14 1026 512 256 1"          # Number of neurons in each layer, including input + output.
acti="None relu relu relu None"   # Activation function for each layer, including input.
                                  # Make sure that the first element (for input) is None.
# Training parameters
E=2          # Total training epoch
batch=100     # Mini-batch size
lrn_rate=0.1  # Learning rate

# Train the model
python -m train_model $out_dir_path $ODIR $ID $R -data_file $data_path  -lrn_rate $lrn_rate -epoch $E -minibatch $batch -neurons $neur -acti $acti -bias -P -norm_const $norm_const
