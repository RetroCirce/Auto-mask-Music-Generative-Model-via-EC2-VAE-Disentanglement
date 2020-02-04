import pretty_midi as pyd


eps = 0.01

def rhythm_loss(x_file,y_file):
    x_midi = pyd.PrettyMIDI(x_file)
    y_midi = pyd.PrettyMIDI(y_file)
    x_notes = [d.start for d in x_midi.instruments[1].notes]
    y_notes = [d.start for d in y_midi.instruments[1].notes]
    # print(x_notes)
    # print(y_notes)
    x_map = [-1] * len(x_notes)
    for i in range(len(x_notes)):
        dis = 1000.0
        for j in range(len(y_notes)):
            tempDis = abs(x_notes[i] - y_notes[j])
            if tempDis < dis and tempDis <= eps:
                x_map[i] = j
                dis = tempDis
    # print(x_map)
    while x_map[-1] == -1:
        x_map.pop()

    prev = -0.1
    p_index = -1
    total_acc = 0.0
    for i in range(len(x_map)):
        if x_map[i] == -1:
            continue
        dis_frac = 0.0
        for j in range(p_index + 1, x_map[i], 1):
            dis_frac += (y_notes[j + 1] - y_notes[j])
        dis_frac /= (x_notes[i] - prev)
        total_acc += (1 - dis_frac)
        prev = x_notes[i]
        p_index = x_map[i]
    total_acc = total_acc / len(x_notes)

    return total_acc
        




total_acc_0 = 0.0
total_acc_1 = 0.0
total_acc_2 = 0.0
total_acc_3 = 0.0

total_acc_4 = 0.0

for i in range(0,2064):
    acc_4 = rhythm_loss("results/o/o_" + str(i) + ".mid", "results/o/o_" + str(i) + ".mid")
    total_acc_4 += acc_4

total_acc_4 /= 2064

for i in range(0,45):
    acc_0 = rhythm_loss("results/o/o_" + str(i) + ".mid", "results/o/o_" + str(i) + ".mid")
    acc_1 = rhythm_loss("results/o/o_" + str(i) + ".mid", "results/v4/v4_" + str(i) + ".mid")
    acc_2 = rhythm_loss("results/o/o_" + str(i) + ".mid", "results/v5/v5_" + str(i) + ".mid")
    acc_3 = rhythm_loss("results/o/o_" + str(i) + ".mid", "results/v7/v7_" + str(i) + ".mid")
    print("Self: %lf | Model_1: %lf | Model_2: %lf | Model_3: %lf" %(acc_0,acc_1,acc_2,acc_3))
    total_acc_0 += acc_0
    total_acc_1 += acc_1
    total_acc_2 += acc_2
    total_acc_3 += acc_3
print((total_acc_0 / 45) / total_acc_4 )
print((total_acc_1 / 45) / total_acc_4 )
print((total_acc_2 / 45) / total_acc_4)
print((total_acc_3 / 45) / total_acc_4)


