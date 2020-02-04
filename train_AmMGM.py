import numpy as np
import os
import shutil
import random
from model_mask_cond import MaskNN
import torch
import sklearn.utils
from torch.nn import functional as F
vector_num = 256
cond_num = 12
rest_pitch = 129
hold_pitch = 128
cpath = "processed_data"
train_path = cpath + "/vae_train_data.npy"
validate_path = cpath + "/vae_validate_data.npy"
test_path = cpath + "/vae_test_data.npy"

total_len = 256 # 32.0 s
known_len = 128 # 16.0 s
shift_len = 32 # 4.0 s
total_epoch = 20000
batch_size = 64
learning_rate = 1e-4
train_data = np.load(train_path,allow_pickle = True)
test_data = np.load(test_path,allow_pickle = True)
validate_data = np.load(validate_path,allow_pickle = True)
print(train_data.shape)
print(test_data.shape)
print(validate_data.shape)
def create_mask(data):
    data_x = []
    data_gd = []
    for x in data:
        temp_gd = x[0].numpy()
        temp_x = np.concatenate((x[0].numpy(),x[1]),axis = 1)
        temp_x[4:,:128] = 0
        data_x.append(temp_x)
        data_gd.append(temp_gd)
#     print(data_x[0])
#     print(data_gd[0])
    return np.array(data_x),np.array(data_gd)
train_x,train_gd = create_mask(train_data)
print(train_x.shape)
print(train_gd.shape)
test_x,test_gd = create_mask(test_data)
print(test_x.shape)
print(test_gd.shape)
validate_x,validate_gd = create_mask(validate_data)
print(validate_x.shape)
print(validate_gd.shape)

# train
model = MaskNN(input_dims = vector_num ,
        hidden_dims = vector_num + 128,output_dims = 128,time_steps = total_len)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(torch.cuda.current_device()),flush = True)
    model.cuda()
else:
    print("Using CPU",flush = True)

model.train()

for epoch in range(total_epoch):
    print("epoch: %d\n_________________________________" % epoch,flush = True)
    train_x, train_gd = sklearn.utils.shuffle(train_x, train_gd)
    train_batches_x = np.split(train_x,
                        range(batch_size, train_x.shape[0] // batch_size * batch_size, batch_size))
    train_batches_gd = np.split(train_gd,
                        range(batch_size, train_gd.shape[0] // batch_size * batch_size, batch_size))
    validate_x, validate_gd = sklearn.utils.shuffle(validate_x, validate_gd)
    validate_batches_x = np.split(validate_x,
                        range(batch_size, validate_x.shape[0] // batch_size * batch_size, batch_size))
    validate_batches_gd = np.split(validate_gd,
                        range(batch_size, validate_gd.shape[0] // batch_size * batch_size, batch_size))
    for i in range(len(train_batches_x)):
        x = torch.from_numpy(train_batches_x[i]).float()
        gd = torch.from_numpy(train_batches_gd[i]).float()
        j = i % len(validate_batches_x)
        v_x = torch.from_numpy(validate_batches_x[j]).float()
        v_gd = torch.from_numpy(validate_batches_gd[j]).float()
        if torch.cuda.is_available():
            x = x.cuda()
            gd = gd.cuda()
            v_x = v_x.cuda()
            v_gd = v_gd.cuda()
        optimizer.zero_grad()
        x_out = model(x)
        loss = F.mse_loss(x_out, gd.float())
        loss.backward()
        optimizer.step()
        v_loss = 0.0
        with torch.no_grad():
            v_x_out = model(v_x)
            v_loss = F.mse_loss(v_x_out, v_gd.float())
        print("batch %d loss: %.5f | val loss %.5f"  % (i,loss.item(),v_loss.item()),flush = True)
    if (epoch + 1) % 100 == 0:
        torch.save(model.cpu().state_dict(), "model_v5_" + str(epoch) + ".pt")
        model.cuda()
