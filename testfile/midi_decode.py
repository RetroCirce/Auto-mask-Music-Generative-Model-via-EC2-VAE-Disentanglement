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


new_outputs= np.load("model_v7_midi.npy")

test_data_real = np.load("test_data_no_split_8_crop.npy")
render = MIDI_Render(datasetName = "Nottingham", minStep= 0.125)

# vae_model = VAE(130, 2048, 3, 12, 128, 128, 32)
# vae_model.eval()
# dic = torch.load("vae/tr_chord.pt")
# for name in list(dic.keys()):
#     dic[name.replace('module.', '')] = dic.pop(name)
# vae_model.load_state_dict(dic)
# if torch.cuda.is_available():
#     vae_model = vae_model.cuda()
for kkk in range(len(new_outputs)):
    # print(kkk)
    # z1 = new_outputs[kkk][0][:,:128]
    # z2 = new_outputs[kkk][0][:,128:]
    # chord_cond = new_outputs[kkk][1]
    # # print(z1.shape)
    # # print(z2.shape)
    # # print(chord_cond.shape)
    # z1 = torch.from_numpy(z1).float()
    # z2 = torch.from_numpy(z2).float()
    # chord_cond = torch.from_numpy(chord_cond).float()
    # # z1 = z1.cuda()
    # # z2 = z2.cuda()
    # # chord_cond = chord_cond.cuda()
    # res = vae_model.decoder(z1,z2,chord_cond)
    # # print(res.detach().cpu().numpy().shape)
    # q = np.asarray((res.detach().cpu().numpy()))
    # temp = []
    # for i in q:
    #     for j in i:
    #         temp.append(np.argmax(j))
    # q = np.array(temp)

    # song = {"notes":q, "chords": test_data_real[kkk]["chords"]}
    # render.data2midi(data = new_outputs[kkk],output = "results/o/o_" + str(kkk) + ".mid")
    render.data2midi(data = test_data_real[kkk],output = "results/o/o_" + str(kkk) + ".mid")