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
train_path = "../dataset/Nottingham/train/"
validate_path = "../dataset/Nottingham/validate/"
test_path = "../dataset/Nottingham/test/"
min_step = 0.125 # for vae
total_len = 256 # 32.0 s
known_len = 128 # 16.0 s
shift_len = 32 # 4.0 s

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
    
def vae_make_one_hot_data(train_data):
    print("convert data to one-hot...",flush = True)
    train_size = min(len(train_data),200)

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
        # for j, value in enumerate(mi):
        #     if j < known_len:
        #         if value != hold_pitch:
        #             prev = value
        #         if value == hold_pitch and mi[j + 1] != hold_pitch:
        #             train_x[i,j,prev] = 1
        #         elif j + 1 == known_len and value == hold_pitch:
        #             train_x[i,j,prev] = 1
        #         else:
        #             train_x[i,j,value] = 1
        # for j, value in enumerate(ci):
        #     train_x[i,j, value + pitch_num] = 1
        # prev = rest_pitch
        # for j, value in enumerate(mi):
        #     if value != hold_pitch:
        #         prev = value
        #     if j + 1 == len(mi):
        #         train_gd[i,j] = prev
        #     elif value == hold_pitch and mi[j + 1] != hold_pitch:
        #         train_gd[i,j] = prev
        #     else:
        #         train_gd[i,j] = value
    print("convert success！",flush = True)
    return [train_x,train_cond]

# train_loader = MIDI_Loader(datasetName = "Nottingham", minStep = min_step)
# validate_loader = MIDI_Loader(datasetName = "Nottingham", minStep = min_step)
test_loader = MIDI_Loader(datasetName = "Nottingham", minStep = min_step)

# train_loader.load(directory = train_path)
# validate_loader.load(directory = validate_path)
test_loader.load(directory = test_path)

# train_loader.getChordSeq()
# validate_loader.getChordSeq()
test_loader.getChordSeq()

# train_loader.getNoteSeq()
# validate_loader.getNoteSeq()
test_loader.getNoteSeq()

# train_data = train_loader.dataAugment()
# validate_data = validate_loader.dataAugment()
test_data = test_loader.dataAugment()

render = MIDI_Render(datasetName = "Nottingham", minStep= min_step)
# for inx,i in enumerate(train_data):
#     print("length: %d" %len(i))
#     render.data2midi(data = i, output = "temp/train_" + str(inx) + ".mid")

# for inx,i in enumerate(validate_data):
#     print("length: %d" %len(i))
#     render.data2midi(data = i, output = "temp/validate_" + str(inx) + ".mid")

# for inx,i in enumerate(test_data):
#     # print("length: %d" %len(i))
#     # render.data2midi(data = i, output = "temp/no_split_test_" + str(inx) + ".mid")
#     if inx % 5 == 0:
#         print("length: %d" %len(i))
#         render.data2midi(data = i, output = "temp/no_split_test_" + str(inx) + ".mid")


# # process data to explicit structure - ont hot vectors

# aligment the data
# train_data = alignment_data(train_data)
# validate_data = alignment_data(validate_data)
test_data = alignment_data(test_data)
te = []
for i in test_data:
    te.append({"notes":i["notes"], "chords": i["chord_seq"]})
# # split the data
# train_data = split_data(train_data,fix_len = total_len,shift_len = shift_len)
# validate_data = split_data(validate_data,fix_len = total_len,shift_len = shift_len)
# test_data = split_data(test_data,fix_len = total_len,shift_len = shift_len)

# tr = np.asarray(train_data)
# va = np.asarray(validate_data)
te = np.asarray(te)

# for inx,i in enumerate(train_data):
#     print("length: %d" %len(i))
#     render.data2midi(data = i, output = "temp/split/train_" + str(inx) + ".mid")

# for inx,i in enumerate(validate_data):
#     print("length: %d" %len(i))
#     render.data2midi(data = i, output = "temp/split/validate_" + str(inx) + ".mid")

for inx,i in enumerate(te):
    if inx % 5 == 0:
        print("length: %d" %len(i))
        render.data2midi(data = i, output = "temp/no_split_test_" + str(inx) + ".mid")
#     print("length: %d" %len(i))
#     render.data2midi(data = i, output = "temp/split/test_" + str(inx) + ".mid")

# print("start to save npy")
# np.save("train_data_8_crop.npy",tr)
# np.save("validate_data_8_crop.npy",va)
np.save("test_data_no_split_8_crop.npy",te)
