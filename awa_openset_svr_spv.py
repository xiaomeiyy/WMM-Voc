import scipy.io as sio
#import word2vec
import h5py
import numpy as np
from scipy.spatial.distance import cdist
import time
import random

## create openset_awa small_dic1000 index---class_id
#dataset = h5py.File('/home/wxm/experiment_wxm/datasets/awa/resnet/awa_resnet_datasets.mat')

# ## proto1000
dic_small = h5py.File('/home/wxm/experiment_wxm/datasets/dictionary/selected_smaller_dic1000.mat','r')
awa_proto = sio.loadmat('/home/wxm/experiment_wxm/datasets/awa/awa_proto1000.mat')
awa_proto = np.array(awa_proto['awa_proto'])
dic = np.array(dic_small['selected_dict1000']).T

##proto100
#dic_small = h5py.File('/home/wxm/experiment_wxm/datasets/dictionary/selected_smaller_dic.mat','r')
#dic_small = sio.loadmat('/home/wxm/experiment_wxm/datasets/dictionary/selected_smaller_dic.mat')
#dic = np.array(dic_small['selected_dict'])
#awa_proto = sio.loadmat('/home/wxm/experiment_wxm/datasets/awa/awa_proto_100.mat')
#awa_proto = np.array(awa_proto['awa_proto'])

distance = cdist(awa_proto,dic)
val_1 = np.sort(distance,axis=1)
idx_1 = np.argsort(distance,axis=1)
cls_id = np.array(idx_1.T[0])

## load data and change test_data id
print('start to compute test_data id ...')
start = time.clock()
awa_feature = h5py.File('/home/wxm/experiment_wxm/datasets/awa/resnet/awa_resnet_datasets.mat')
y_te = np.array(awa_feature['y']['teS'])
x_te = np.array(awa_feature['x']['teS']).T
awa_feature.close()
print(y_te.shape)
print(x_te.shape)
y_te2 = np.zeros_like(y_te)
for i in range(len(cls_id)):
    idx_f = np.argwhere(y_te == (i+1))
    #y_te2[idx_f-1] = cls_id[i]
    y_te2[idx_f[:, 0]] = cls_id[i]
# write awa_proto id to .mat file
#dataNew = 'awa_resnet_datasets_y_te_proto100.mat'
#sio.savemat(dataNew, {'y_te_proto100': y_te2})
elapsed = (time.clock() - start)
print('end of computing test_data id, time used :', elapsed)

## compute distance awa_proto and dic,sort distance
w = sio.loadmat('w_awa_svr.mat')
w = np.array(w['w'])
embed = x_te.dot(w)
print('start distance')
distance_2 = cdist(embed,dic)
print('end distance')
val_2 = np.sort(distance_2,axis=1)
idx_2 = np.argsort(distance_2,axis=1)
#write to h5 file
val_3 = val_2[:,0:100]
idx_3 = idx_2[:,0:100]
file_h5_write = h5py.File('w_awa_svr_distance2_spv.h5','w')
file_h5_write.create_dataset('distance', data=distance_2)
file_h5_write.create_dataset('val', data=val_3)
file_h5_write.create_dataset('idx', data=idx_3)
#print(val_2)
#print(idx_2)

## compute openset zsl accuracy
## load data : distance and y_te
idx = idx_2.T
#y = sio.loadmat('awa_resnet_datasets_y_te_proto100.mat')
y_te = y_te2

## caculate prediction
print('start to caculate prediction ...')
start = time.clock()
for hist in range(50):
    Y_hist = idx[0:hist+1].T
    n = 0
    for i in range(len(y_te)):
        y_te_i = int(y_te[i])
        if y_te_i in Y_hist[i]:
            n = n + 1
    accuracy = n / len(y_te)
    print('hist', hist + 1, " accuracy: %.4f" % (accuracy * 100))
elapsed = (time.clock() - start)
print('end to caculate prediction, time used :', elapsed)