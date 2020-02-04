from dataloader import MIDI_Loader,MIDI_Render
import numpy as np
import os
import shutil
import random
from model import AutoDropoutNN
import torch
import sklearn.utils
from torch.nn import functional as F

pitch_num = 130
chord_num = 25
rest_pitch = 128
hold_pitch = 129
none_chord = 24
recog_level = "Mm"
train_path = "../dataset/Nottingham/train/"
validate_path = "../dataset/Nottingham/validate/"
test_path = "../dataset/Nottingham/test/"
min_step = 0.03125
total_len = 640 # 20.0 s
known_len = 320 # 10.0 s
shift_len = 128 # 5.0 s
total_epoch = 100
batch_size = 32
learning_rate = 1e-3

def split_dataset(directory, rate = [0.6,0.8,1.0]):
    path = os.listdir(directory)
    random.shuffle(path)
    nums = len(path)
    train_files = path[:int(nums * rate[0])]
    vali_files = path[int(nums * rate[0]):int(nums * rate[1])]
    test_files = path[int(nums * rate[1]):int(nums * rate[2])]
    for i in train_files:
        shutil.copyfile(directory + i, "dataset/Nottingham/train/" + i)
        print("copy %s success!\n" %i)
    for i in vali_files:
        shutil.copyfile(directory + i, "dataset/Nottingham/validate/" + i)
        print("copy %s success!\n" %i)
    for i in test_files:
        shutil.copyfile(directory + i, "dataset/Nottingham/test/" + i)
        print("copy %s success!\n" %i)

def alignment_data(data):
    new_data = []
    alignment_num = 0
    for e in data:
        delta = len(e["notes"]) - len(e["chord_seq"])
        if delta < 0:
            if e["notes"] == []:
                continue
            if e["notes"][-1] == rest_pitch or e["notes"][-1] == hold_pitch:
                q = e["notes"][-1]
                alignment_num += 1
                for i in range(-delta):
                    e["notes"].append(q)
            elif 0 <= e["notes"][-1] <= 127:
                q = hold_pitch
                alignment_num += 1
                for i in range(-delta):
                    e["notes"].append(q)
        elif delta > 0:
            if e["chord_seq"] == []:
                continue
            q = e["chord_seq"][-1]
            alignment_num += 1
            for i in range(delta):
                e["chord_seq"].append(q)
        new_data.append(e)
    print("finished %d data, %d data need aligment" %(len(data),alignment_num))
    return new_data

def split_data(data, fix_len = 640, shift_len = 128):
    new_data = []
    print("begin split_data")
    for i, d in enumerate(data):
        if i % 500 == 0:
            print("finish %d data" %i)
        mi = d["notes"]
        ci = d["chord_seq"]
        sta_pos = 0
        while ci[sta_pos] == none_chord:
            sta_pos += 1
        for j in range(sta_pos, len(ci) - 2 * fix_len, shift_len):
            split_sta = j
            split_flag = False
            while 1 == 1:
                if split_sta >= j + shift_len:
                    break
                if (ci[split_sta] != none_chord and 
                    ci[split_sta] != ci[split_sta - 1] and 
                    mi[split_sta] != hold_pitch and 
                    mi[split_sta] != rest_pitch):
                    split_flag = True
                    break
                split_sta += 1
            if not split_flag:
                continue
            split_end = -1
            for k in range(split_sta + fix_len - shift_len , split_sta + fix_len):
                if ((mi[k] == hold_pitch or mi[k] == rest_pitch) and 
                    mi[k + 1] != hold_pitch and
                    ci[k] != ci[k + 1]):
                    split_end = k
            if split_end == -1:
                continue
            split_end += 1
            n_m = d["notes"][split_sta:split_end]
            n_c = d["chord_seq"][split_sta:split_end]
            if fix_len - split_end + split_sta > 0:
                for i in range(fix_len - split_end + split_sta):
                    n_m.append(rest_pitch)
                    n_c.append(none_chord)
            new_data.append({"notes": n_m, "chords": n_c})
    print("finished %d data, %d split data get" %(len(data),len(new_data)))
    return new_data
    
def make_one_hot_data(train_data):
    print("convert data to one-hot...")
    train_size = min(len(train_data),5000)

    train_x = np.zeros((train_size,total_len,pitch_num + chord_num), dtype = np.int32)
    train_gd = np.zeros((train_size,total_len), dtype = np.int64)


    # process with bi-directional issue

    for i,data in enumerate(train_data):
        if i >= train_size:
            break
        mi = data["notes"]
        ci = data["chords"]
        prev = rest_pitch
        for j, value in enumerate(mi):
            if j < known_len:
                if value != hold_pitch:
                    prev = value
                if value == hold_pitch and mi[j + 1] != hold_pitch:
                    train_x[i,j,prev] = 1
                elif j + 1 == known_len and value == hold_pitch:
                    train_x[i,j,prev] = 1
                else:
                    train_x[i,j,value] = 1
        for j, value in enumerate(ci):
            train_x[i,j, value + pitch_num] = 1
        prev = rest_pitch
        for j, value in enumerate(mi):
            if value != hold_pitch:
                prev = value
            if j + 1 == len(mi):
                train_gd[i,j] = prev
            elif value == hold_pitch and mi[j + 1] != hold_pitch:
                train_gd[i,j] = prev
            else:
                train_gd[i,j] = value
    print("convert success！")
    return [train_x,train_gd]

# def train():
   
# # load data from three folders
# train_loader = MIDI_Loader(datasetName = "Nottingham", minStep = min_step)
# validate_loader = MIDI_Loader(datasetName = "Nottingham", minStep = min_step)
# test_loader = MIDI_Loader(datasetName = "Nottingham", minStep = min_step)

# train_loader.load(directory = train_path)
# validate_loader.load(directory = validate_path)
# test_loader.load(directory = test_path)

# train_loader.getChordSeq()
# validate_loader.getChordSeq()
# test_loader.getChordSeq()

# train_loader.getNoteSeq()
# validate_loader.getNoteSeq()
# test_loader.getNoteSeq()

# train_data = train_loader.dataAugment()
# validate_data = validate_loader.dataAugment()
# test_data = test_loader.dataAugment()

# # process data to explicit structure - ont hot vectors

# # aligment the data
# train_data = alignment_data(train_data)
# validate_data = alignment_data(validate_data)
# test_data = alignment_data(test_data)

# # split the data
# train_data = split_data(train_data,fix_len = total_len,shift_len = shift_len)
# validate_data = split_data(validate_data,fix_len = total_len,shift_len = shift_len)
# test_data = split_data(test_data,fix_len = total_len,shift_len = shift_len)

# tr = np.asarray(train_data)
# va = np.asarray(validate_data)
# te = np.asarray(test_data)

# print("start to save npy")
# np.save("train_data.npy",tr)
# np.save("validate_data.npy",va)
# np.save("test_data.npy",te)
# print("finish saving npy")
# # render the data to files
# # render = MIDI_Render(datasetName = "Nottingham",minStep= min_step)
# # for i,v in enumerate(train_data):
# #     if i > 2000:
# #         break
# #     render.data2midi(data = v, recogLevel = "Mm", output = "splited/train/" + str(i) + ".mid")
# # for i,v in enumerate(test_data):
# #     if i > 2000:
# #         break
# #     render.data2midi(data = v, recogLevel = "Mm", output = "splited/test/" + str(i) + ".mid")
# # for i,v in enumerate(validate_data):
# #     if i > 2000:
# #         break
# #     render.data2midi(data = v, recogLevel = "Mm", output = "splited/validate/" + str(i) + ".mid")
  
# # process train_x train_gd validate_x validate_gd
# # convert sequence data to one-hot vectors
train_data = np.load("train_data.npy")
test_data = np.load("test_data.npy")
validate_data = np.load("validate_data.npy")

train_x,train_gd = make_one_hot_data(train_data)
test_x,test_gd = make_one_hot_data(test_data)
validate_x,validate_gd = make_one_hot_data(validate_data)

print(train_x.shape)
print(train_gd.shape)

# train
model = AutoDropoutNN(input_dims = pitch_num + chord_num,
        hidden_dims = 2 * (pitch_num + chord_num),output_dims = pitch_num,time_steps = total_len)

optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print("Using CPU")

# from torchsummary import summary

# summary(model, input_size=(640, 155))
    
model.train()
for epoch in range(total_epoch):
    print("epoch: %d\n_________________________________" % epoch)
    train_x, train_gd = sklearn.utils.shuffle(train_x, train_gd)
    train_batches_x = np.split(train_x,
                        range(batch_size, train_x.shape[0] // batch_size * batch_size, batch_size))
    train_batches_gd = np.split(train_gd,
                        range(batch_size, train_gd.shape[0] // batch_size * batch_size, batch_size))
    for i in range(len(train_batches_x)):
        with torch.autograd.set_detect_anomaly(True):
            x = torch.from_numpy(train_batches_x[i]).float()
            gd = torch.from_numpy(train_batches_gd[i]).float()
            if torch.cuda.is_available():
                x = x.cuda()
                gd = gd.cuda()
            optimizer.zero_grad()
            x_out = model(x)
            loss = F.cross_entropy(x_out.view(-1,x_out.size(-1)), gd.view(-1).long())
            loss.backward()
            optimizer.step()
            print("batch %d loss: %.5f" % (i,loss.item()) )
torch.save(model.cpu().state_dict(), "test_model.pt")
model.cuda()