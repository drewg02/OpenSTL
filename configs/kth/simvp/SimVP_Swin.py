method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'swin'
hid_S = 64
hid_T = 256
N_T = 6
N_S = 2
# training
lr = 5e-4
drop_path = 0.1
batch_size = 4  # bs = 4 x 4GPUs
sched = 'onecycle'