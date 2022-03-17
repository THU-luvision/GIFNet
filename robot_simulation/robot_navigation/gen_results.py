import numpy as np
import random as rnd
import os.path as osp
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib
import json
def json_file_load(file_path):
    if not os.path.exists(file_path):
        print('Can not find: ', file_path)
        raise FileNotFoundError
    with open(file_path, 'r') as load_f:
        json_object = json.load(load_f)
    return json_object


if __name__ == '__main__':
    pi_t = []
    pe_t = []
    for dd in range(10):
        savename = 'data/datas' + str(dd)

        x = ['Direct-S','Direct-L','GIF']
        x_t = ["1v1","1v2","1v7"]
        num_of_people = [2,3,4,8]
        x_range = np.arange(0,3,1)
        y_pe = []
        y_pi = []
        y_pe_robo = []
        y_pi_robo = []
        y_pe_total = []
        y_pi_energy = []
        for i in [0,2,3]:
            filename = 'unl-'+str(i+2)+'-astar-double'
            datas = json_file_load(os.path.join(savename,filename,str(i)))
            pe_no_robo,pi_no_robo,eng_no_robo = datas['no_robo'][2], datas['no_robo'][3],datas['no_robo'][4]
            robo_pe,robo_pi, pe,pi,eng = datas['180-20'][0][0],datas['180-20'][1][0],datas['180-20'][2], datas['180-20'][3],datas['180-20'][4]

            robo_pe1,robo_pi1,pe1,pi1,eng1 = datas['180-3'][0][0],datas['180-3'][1][0],datas['180-3'][2], datas['180-3'][3],datas['180-3'][4]
            robo_pe2,robo_pi2,pe2,pi2,eng2 =datas['60-20'][0][0],datas['60-20'][1][0], datas['60-20'][2], datas['60-20'][3],datas['60-20'][4]
            
            y_de_pe = [(pe1-pe_no_robo)/pe_no_robo*100,(pe-pe_no_robo)/pe_no_robo*100,(pe2-pe_no_robo)/pe_no_robo*100]
            y_de_pi = [(pi1-pi_no_robo),(pi-pi_no_robo),(pi2-pi_no_robo)]
            eng_no_robo = np.sum(eng_no_robo) / num_of_people[i]
            eng = np.mean(eng[1:])*(num_of_people[i]-1)/num_of_people[i]+eng[0]/num_of_people[i]
            eng1 = np.mean(eng1[1:])*(num_of_people[i]-1)/num_of_people[i]+eng1[0]/num_of_people[i]
            eng2 = np.mean(eng2[1:])*(num_of_people[i]-1)/num_of_people[i]+eng2[0]/num_of_people[i]
            y_eng = [eng1-eng_no_robo,eng-eng_no_robo,eng2-eng_no_robo]
            y_pi_energy.append(y_eng)
            y_pe_robo.append([robo_pe1-1,robo_pe-1,robo_pe2-1])
            y_p_t = [100 + y_de_pe[0]*(num_of_people[i]-1)/num_of_people[i]+(robo_pe1-1)/num_of_people[i]*100,
                    100 + y_de_pe[1]*(num_of_people[i]-1)/num_of_people[i]+(robo_pe-1)/num_of_people[i]*100, 
                    100 +y_de_pe[2]*(num_of_people[i]-1)/num_of_people[i]+(robo_pe2-1)/num_of_people[i]*100]
            y_pe_total.append(y_p_t)
            y_pi_robo.append([robo_pi1,robo_pi,robo_pi2])
            y_pe.append(y_de_pe)
            y_pi.append(y_de_pi)

        y_pe =np.array(y_pe)
        y_pi =np.array(y_pi)
        y_pe_robo = np.array(y_pe_robo)
        y_pi_robo = np.array(y_pi_robo)
        y_pe_total = np.array(y_pe_total)
        y_pi_energy = np.array(y_pi_energy)

        pe_t.append(y_pe_total)
        pi_t.append(y_pi_energy)



    pe_t = np.array(pe_t)
    pi_t = np.array(pi_t)
    pe_std = np.std(np.mean(pe_t,1),0)
    pi_std = np.std(np.mean(pi_t,1),0)
    y_pe_total = np.mean(pe_t,0)
    y_pi_energy = np.mean(pi_t,0)

    print(pe_std)
    print(pi_std)
    print(np.mean(y_pi_energy,0))
    print(np.mean(y_pe_total,0))

