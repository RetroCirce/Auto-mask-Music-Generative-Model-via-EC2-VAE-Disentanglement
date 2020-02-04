import torch
import pretty_midi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model import VAE
from utils import numpy_to_midi_with_condition
import math
import decimal

rest_state = 129

none_chord = 24
hold_chord = 25
chord_stage = 4
chord_order = [[0,4,7],[0,3,7]]

def gen_chord(chord_num):
    if chord_num == none_chord:
        return []
    pitch_pos = int(chord_num / 2)
    type_pos = int(chord_num % 2)
    re = []
    for i in range(3):
        re.append((chord_order[type_pos][i] + pitch_pos) % 12)
    return re

def sSigmoid(x):
    return round(float(1.0 - (1.0 / (1.0 + math.exp(-0.05 * x)) - 0.5) * 2.0),2)
def compareS(vec, gd):
    similarity = 99999.0
    for value in gd:
        out = 0.0
        for valueIndex in range(0, len(value)):
            out = out + math.pow(value[valueIndex] - vec[valueIndex], 2)
        similarity = min(similarity, out)
    return sSigmoid(similarity)#round(float(similarity),2)


def plotGen(sample, gds, index):
    mat = []
    for i in gds:
        simPerSong = []
        for j in range(0,len(sample)):
            simPerSong.append(0.0)
        mat.append(simPerSong)
    mat = np.array(mat)
    for i, gd in enumerate(gds):
        for j, vec in enumerate(sample):
            mat[i, j] = compareS(vec, gd)
    #print(mat)

    fig, ax = plt.subplots()
    im = ax.imshow(mat)
    ax.tick_params(labelsize = 6)
    ax.set_xticks(np.arange(len(sample)))
    ax.set_yticks(np.arange(len(gds)))
    ax.set_xticklabels(np.arange(len(sample)))
    ax.set_yticklabels(np.arange(len(gds)))
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

    for i in range(len(gds)):
        for j in range(len(sample)):
            text = ax.text(j, i, mat[i, j], ha = "center", va = "center", color = "w", fontsize = 6)

    fig.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(20,140)
    fig.savefig('figure3/' + str(index) + '.png', dpi = 100)
    #plt.show()



# initialize
model = VAE(130, 2048, 3, 12, 128, 128, 32)
model.eval()
dic = torch.load("params/tr_chord.pt")
for name in list(dic.keys()):
    dic[name.replace('module.', '')] = dic.pop(name)
model.load_state_dict(dic)
if torch.cuda.is_available():
    model = model.cuda()

print(model)


# from data
sample_train = np.load("vae_sample_v7_train.npy",  encoding="latin1")
sample_test = np.load("vae_sample_v8_test.npy",  encoding="latin1")
sample_condition = np.load("vae_data/chord_dataset.npy",  encoding="latin1")
skipIndex = np.load("melody_test_index.npy",encoding="latin1")


# print(len(sample))
# output_sample = []
# output_sample_con = []
# for sm in sample:
#     output_sm = []
#     output_smm = np.zeros((32, 130), dtype = np.int32)
#     t = 0
#     for smm in sm:
#         output_smm[t, smm] = 1
#         t = t + 1
#         if t == 32:
#             t = 0
#             output_sm.append(output_smm)
#             output_smm = np.zeros((32, 130), dtype = np.int32)
#     #print(len(output_sm))
#     output_sample.append(output_sm)

output_sample_con = []

for cm in sample_condition:
    output_cm = []
    output_cmm = np.zeros((32, 12), dtype = np.int32)
    t = 0
    cur_cmm = 0
    for cmm in cm:
        if cmm[0] != hold_chord:
          cur_cmm = cmm[0]
        chord_vec = gen_chord(cur_cmm)
        for ci in chord_vec:
            output_cmm[t,ci] = 1
        t = t + 1
        if t == 32:
            t = 0
            output_cm.append(output_cmm)
            output_cmm = np.zeros((32, 12), dtype = np.int32)
    output_sample_con.append(output_cm)
# ---------------
condition_train = []
condition_test = []
for i,value in enumerate(output_sample_con):
    flag = 0
    for j in skipIndex:
        if i == j:
            flag = 1
            break
    if flag == 0:
        condition_train.append(value)
    else:
        condition_test.append(value)

o_condition_train = []
o_condition_test = []
for i,value in enumerate(sample_condition):
    flag = 0
    for j in skipIndex:
        if i == j:
            flag = 1
            break
    if flag == 0:
        o_condition_train.append(value)
    else:
        o_condition_test.append(value)


x = sample_train[0]
z1 = x[:,0:128]
z2 = x[:,128:256]
print(z1.shape)
print(z2.shape)
# gen data

for i in range(len(sample_test)):
    # print(sample.shape)
    # sm = torch.from_numpy(sample_train[select_index]).float()
    x = sample_test[i]
    splitDis = min(len(x),len(condition_test[i]))
    cm = np.array(condition_test[i])
    cm = cm[:splitDis]
    cm = torch.from_numpy(cm).float()
    z1 = x[:splitDis,0:128]
    z2 = x[:splitDis,128:256]
    z1 = torch.from_numpy(z1).float()
    z2 = torch.from_numpy(z2).float()
    output = model.decoder(z1, z2, cm)
    print(output.shape)
    numpy_to_midi_with_condition(output, o_condition_test[i], "gen_v8_test/vae_origin_" + str(i) + ".mid")