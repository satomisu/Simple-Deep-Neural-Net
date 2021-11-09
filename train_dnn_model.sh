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
# Normalization constant for the label
norm_const=311.17   # Arbitrarily picked from Kuka's sv

# Network parameters
neur="14 1026 512 256 1"
acti="None relu relu relu None"

# Training parameters
E=200         # Epochs
batch=100 # Mini-batch size
lrn_rate=0.1  # Learning rate

# Train the model
python -m train_model $ODIR $ID $R -dfile $data_path -out_dir_path $out_dir_path -lrn_rate $lrn_rate -epoch $E -minibatch $batch -neurons $neur -acti $acti -bias -P -save_eval_at_each_epoch -norm_const $norm_const
