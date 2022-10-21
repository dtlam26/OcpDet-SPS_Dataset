import sys
from utils import non_max_suppression,swap_xy,draw_location,show
import cv2
from trt_model import *
import numpy as np
import time
import matplotlib.pyplot as plt

model = TrtModel('spatial_FP16.engine')
mean = [0.485*255, 0.456*255, 0.406*255]
scale = [255,255,255]
training_mean, training_std = [7.795723324128529, 3.7053501571251233]

aa_keys=[
    'ssd_mobile_net_v2fpn_learning_loss_keras_feature_extractor/Sigmoid:0',
    'ssd_mobile_net_v2fpn_learning_loss_keras_feature_extractor/Sigmoid_1:0',
    'ssd_mobile_net_v2fpn_learning_loss_keras_feature_extractor/Sigmoid_2:0',
    'ssd_mobile_net_v2fpn_learning_loss_keras_feature_extractor/Sigmoid_3:0'
]
ll_key = 'ssd_mobile_net_v2fpn_learning_loss_keras_feature_extractor/learning_loss_module/ll_pred/MatMul:0'
s_key = 'Postprocessor/convert_scores:0'
b_key = 'Postprocessor/Squeeze:0'

def unpad_from_square(xyxy,w,h):
    if h>w:
        xyxy[:,:4:2] = (xyxy[:,:4:2]-0.5)*h/w+0.5
    elif w>h:
        xyxy[:,1:4:2] = (xyxy[:,1:4:2]-0.5)*w/h+0.5
    return xyxy

#Inference image
image_path = 'test_imgs/3.png'
im = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
h,w,c = im.shape
im_batch = np.expand_dims(model.preprocess(im,(896,896),mean=mean,stddev=scale,zero_pad=[int((w-h)/2),int((w-h)/2),0,0]),0)
st = time.time()
output = model(im_batch)
print('Forwarding Time:', time.time()-st)

bbs = output[b_key]
scores = output[s_key][:,:,:-2]

reshape_prediction_tensors = []
for i,aa_key in enumerate(aa_keys):
    aa = output[aa_key]
#     shape = aa.shape
#     current_num_anchors = shape[1]*shape[2]*shape[3]
#     if i < 2:
#         aa = np.zeros_like(aa)
    reshape_prediction_tensors.append(aa.reshape(-1,1))
anchor_candidates = np.concatenate(reshape_prediction_tensors,0).reshape(-1)
anchor_candidates[np.where(anchor_candidates<0.5)] = -0.3
mask = np.where(anchor_candidates>0.6)[0].tolist()

full_anchor_candidates = np.stack([anchor_candidates]*6,1).reshape(-1,1)
scores += full_anchor_candidates
scores = np.exp(scores)/np.expand_dims(np.sum(np.exp(scores),-1),-1)


prediction = np.concatenate([bbs,scores],2)[0]
ll = abs(round(float((output[ll_key]-training_mean)/training_std),2))
o,k = non_max_suppression(prediction,conf_thres=0.5 if ll<1 else 0.35,iou_thres=0.35,object_score=False)
o[:,:4] = swap_xy(o[:,:4])
o = unpad_from_square(o,w,h)
print("Learning Loss at", output[ll_key],':', ll, "with", o.shape)

draw_img = im.copy()
draw_location(draw_img,o.tolist(),{1:'occupied',2:'available',3:'illegal',4:'restricted'},width=w,height=h,shape='rect')
show(draw_img)

#Activation Accumulation
#Uncomment this section if you want to observe
# \\\ Start: Anchor activation visualization
spatial_score = []
anchor_per_head = [112*112,56*56,28*28,14*14]

size = [8,16,32,64]
num_anchor_per_loc = 1
pts = []
for m in mask:
    for i in range(len(anchor_per_head)):
        if i == 0:
            if m <= anchor_per_head[i]:
                index = m
                break
        else:
            if m >= sum(anchor_per_head[:i]) and m <= sum(anchor_per_head[:i+1]):
                index = m-sum(anchor_per_head[:i])
                break
    if i <2:
        continue

    y = int(index//(896/size[i]*num_anchor_per_loc))
    x = int((index-(896/size[i])*num_anchor_per_loc*y)//num_anchor_per_loc)
    spatial_score.append(output[aa_keys[i]][0,y//6,x//6])
    pts.append([((x-2)*size[i],(y-2)*size[i]),((x+2)*size[i],(y+2)*size[i])])
pts = np.asarray(pts).astype(np.int32).tolist()
image = cv2.resize(im,(896,896))

plt.figure(figsize=(5,10))
plt.imshow(image, 'gray', interpolation='none')

total = None
for pt in pts:
    base = np.zeros_like(image)
    cv2.rectangle(base, tuple(pt[0]),tuple(pt[1]),color=(1,1,1), thickness=-1)
    if total is None:
        total = base
    else:
        total += base
base = np.zeros_like(image)
for _o in o.tolist():
    base[int(_o[1]*896):int(_o[3]*896),int(_o[0]*896):int(_o[2]*896)] = 1

plt_img = plt.imshow(total[:,:,0]/np.max(total), 'jet', interpolation='none', alpha=0.7)
plt.show()
# \\\ End Anchor activation visualization
