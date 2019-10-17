import matplotlib.pyplot as plt
import numpy as np

def norm(a):
    a = np.array(a)
    a_min = np.min(a)
    a_max = np.max(a)
    a = (a - a_min) / (a_max - a_min)
    a = list(a)
    return a

x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y_100_solo_algin = [0.8193169830736347, 0.8402917323292142, 0.8478295247721787, 0.8505666654514737, 0.8456222721655713,
                    0.8367256910172849, 0.8275414114797608, 0.8081283126656725, 0.7963132223574372, 0.7540009478114189]
y_100_1_8 = [0.14189552509461176, 0.2219112741865353, 0.3294435140228502, 0.42920970740051967, 0.507937686746937,
     0.5639056061906909, 0.6157520012523869, 0.6799139801487633, 0.7046873553128957, 0.7706858755361132]
y_100_raw = [0.12925413620071766, 0.19101782786444355, 0.2725500092714436, 0.3479363809391548, 0.4043539311180042,
             0.4408775243563678, 0.4761639383804491, 0.519660796768504, 0.5385981758441499, 0.5933634067842712]
y_100_solo_algin = y_100_solo_algin
y_100_1_8 = y_100_1_8
y_100_raw = y_100_raw
# plt.plot(x, y_100_solo_algin, 'r')
# plt.plot(x, y_100_1_8, 'b')
plt.plot(x, y_100_raw, 'g')
plt.savefig('./val100.jpg')