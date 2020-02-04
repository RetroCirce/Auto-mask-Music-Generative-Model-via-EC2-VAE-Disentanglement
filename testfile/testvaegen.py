import numpy as np
import os
import shutil
import random
import torch
import sklearn.utils

from vae.model import VAE
# from gen_model.model import AutoDropoutNN
from loader.dataloader import MIDI_Loader,MIDI_Render
from loader.chordloader import Chord_Loader
from torch.nn import functional as F

pitch_num = 130
chord_num = 12
rest_pitch = 129
hold_pitch = 128
none_chord = 24
recog_level = "Mm"
min_step = 0.125 # for vae
total_len = 256 # 32.0 s
known_len = 128 # 16.0 s
shift_len = 32 # 4.0 s


def vae_make_one_hot_data(train_data):
    print("convert data to one-hot...",flush = True)
    train_size = min(len(train_data),1001)

    train_x = np.zeros((train_size,total_len,pitch_num), dtype = np.int32)
    train_cond = np.zeros((train_size,total_len,chord_num), dtype = np.int32)
    # train_gd = np.zeros((train_size,total_len), dtype = np.int32)

    cl = Chord_Loader()
    # process with bi-directional issue

    for i,data in enumerate(train_data):
        if i >= train_size:
            break
        mi = data["notes"]
        ci = data["chords"]
        prev = rest_pitch
        for j,value in enumerate(mi):
            train_x[i, j, value] = 1
        for j,value in enumerate(ci):
            cname = cl.index2name(j)
            cnotes = cl.name2note(cname)
            # print(cnotes)
            if cnotes is None:
                continue
            for k in cnotes:
                train_cond[i,j,k % 12] = 1
    print("convert success！",flush = True)
    return [train_x,train_cond]

train_data = np.load("processed_data/vae_nottingham/train_data_8_crop.npy",allow_pickle=True)
test_data = np.load("processed_data/vae_nottingham/test_data_8_crop.npy",allow_pickle=True)
validate_data = np.load("processed_data/vae_nottingham/validate_data_8_crop.npy",allow_pickle=True)

train_x,train_cond = vae_make_one_hot_data(train_data)
test_x,test_cond = vae_make_one_hot_data(test_data)
validate_x,validate_cond = vae_make_one_hot_data(validate_data)

print(train_x.shape)
print(train_cond.shape)

sindex = 999
a = train_x[sindex]
b = train_cond[sindex]
print(a.shape)
print(b.shape)
a = np.reshape(a, (-1, 32, 130))
b = np.reshape(b, (-1, 32, 12))
print(a.shape)
print(b.shape)
#initialize model
model = VAE(130, 2048, 3, 12, 128, 128, 32)
model.eval()
dic = torch.load("vae/tr_chord.pt")
for name in list(dic.keys()):
    dic[name.replace('module.', '')] = dic.pop(name)
model.load_state_dict(dic)
if torch.cuda.is_available():
    model = model.cuda()
print(model)

a = torch.from_numpy(a).float()
b = torch.from_numpy(b).float()
res = model.encoder(a,b)
z1 = res[0].loc.detach().numpy()
z2 = res[1].loc.detach().numpy()
print(z1)
print(z1.shape)
print(z2)
print(z2.shape)

z1 = torch.from_numpy(z1).float()
z2 = torch.from_numpy(z2).float()
res = model.decoder(z1,z2,b)
print(res.detach().numpy().shape)
q = np.asarray((res.detach().numpy()))
temp = []
for i in q:
    for j in i:
        temp.append(np.argmax(j))
q = np.array(temp)
print(q)
song = {"notes":q, "chords": train_data[sindex]["chords"]}

render = MIDI_Render(datasetName = "Nottingham", minStep= min_step)
render.data2midi(data = song,output = "test.mid")
render.data2midi(data = train_data[sindex],output = "real.mid")
