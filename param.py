# Coded by Xiaokun Liang
# E-mail: xiaokun@qq.com
# Stanford University


# the index of training GPU
gpu_index = '0'

# the dir of the data
training_data_path = './cv_dataset/training'
validation_data_path = './cv_dataset/validation'
testing_data_path = './cv_dataset/testing'

# the dir of the saved model
model_dir = './model_Result'

# number of the training iteration
# num_iter = 100000
num_iter = 10

# the initial learning rate
lr = 0.01

# the decay rate of the learning rate
decay_rate = 0.1

# decay times of the learning rate in training
learning_rate_decay_times = 4

# training batch size
batch_size = 1

# the weight of control volumes' ncc
lambda_cv = 0.5

# the size of the control volume (pixel)
control_volume_size = (30, 40, 40)

# save the model per # iteration
# save_net_nIter = 50
save_net_nIter = 5