import tensorflow as tf
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import pprint as pp
import json
from loguru import logger


COLORS = (cm.get_cmap('Dark2', 8).colors[:,:3]*255).astype(np.int32).tolist()

class VariantParkingSlot_Detector(object):
    def __init__(self,saved_model_path='./weights/spatial/saved_model',category_index={}):
        self.model = tf.saved_model.load(
            saved_model_path, tags=None, options=None
        )

        self.category_index = category_index

    def single_image_output_visualization(self,img,output,format='rect'):
        plot_img = img.copy()
        if isinstance(output,np.ndarray):
            boxes = output.tolist()
        else:
            boxes = output
        for i,(box) in enumerate(boxes):
            label = ''
            if self.category_index:
                label += category_index[box[4]]
                label += ': '
            label += str(round(box[5],2))
            labelSize,_ =cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,1,6)

            if format == 'rect':
                cv2.rectangle(plot_img,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),COLORS[int(box[4])],2)
            elif format == 'circ':
                cv2.circle(plot_img,(int((box[3]+box[1])/2),int((box[2]+box[0])/2)),int((box[3]-box[1])*1/5),COLORS[int(box[4])],2)

            cv2.rectangle(plot_img,
                        (int((box[1]+box[3]-labelSize[0])/2)-5,int((box[0]+box[2]-labelSize[1])/2)-5),
                        (int((box[1]+box[3]+labelSize[0])/2)+5,int((box[0]+box[2]+labelSize[1])/2)+5),
                        color=COLORS[int(box[4])],thickness=-1)

            cv2.putText(plot_img,label, (int((box[1]+box[3]-labelSize[0])/2),int((box[0]+box[2]+labelSize[1])/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color = (255, 255, 255),thickness=2)

        plt.figure(figsize=(15,15))
        plt.imshow(plot_img)

    def __call__(self,image,visualize=False,threshold=0.4):
        if isinstance(image,str):
            image = cv2.imread(image)

        elif isinstance(image,np.ndarray):
            assert image.shape[-1]==3, "Only 3 channel Image is accepted"
            if len(image.shape)!=3:
                raise ValueError("Batch >1 inference is not supported!")
                # img_batch = image
        elif isinstance(image,list):
            # img_batch = np.asarray(image)
            raise ValueError("Batch >1 inference is not supported!")
        h,w,c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_batch = np.expand_dims(image,0)
        output = self.model(img_batch)
        output = {x:y.numpy()[0] for x,y in output.items()}
        additional_output = {}
        # print(output.keys())
        if "learning_loss" in output:
            additional_output['learning_loss'] = float(output['learning_loss'].reshape([]))

        active_anchors = []
        # for k in output:
        #     if 'active_anchors' in k:
        #         active_anchors.append(output[k])
        # if active_anchors:
        #     additional_output['active_anchors'] = active_anchors

        selected = np.where(output['detection_scores']>threshold)

        # output['detection_keypoints'][...,0] = np.clip(output['detection_keypoints'][...,0],0,1)*h
        # output['detection_keypoints'][...,1] = np.clip(output['detection_keypoints'][...,1],0,1)*w

        output['detection_boxes'][:,1:4:2] = output['detection_boxes'][:,1:4:2]*w
        output['detection_boxes'][:,:4:2] = output['detection_boxes'][:,:4:2]*h

        bboxes = output['detection_boxes'][selected].reshape(-1,4)
        classes = output['detection_classes'][selected].reshape(-1,1)
        scores = output['detection_scores'][selected].reshape(-1,1)

        output = np.concatenate([bboxes,classes,scores],axis=-1)
        # output = self.remove_high_anchor_overlap(output)

        if visualize:
            self.single_image_output_visualization(image,output)
        return output, additional_output

    def remove_high_anchor_overlap(self,output, iou_thresh=0.7, keep_top_k=-1):
        if len(output) == 0: return output

        pick = []
        xmin = output[:, 0]
        ymin = output[:, 1]
        xmax = output[:, 2]
        ymax = output[:, 3]
        confidences = output[:, -1]
        area = (xmax - xmin + 1e-6) * (ymax - ymin + 1e-6)
        area_sort_id = np.argsort(area)
        confidences[area_sort_id] = confidences[area_sort_id] + 1e-6*np.arange(len(area), 0, -1)
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # keep top k
            if keep_top_k != -1:
                if len(pick) >= keep_top_k:
                    break

            overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
            overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
            overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
            overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
            overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
            overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
            overlap_area = overlap_w * overlap_h
            # overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)
            overlap_ratio = overlap_area / (area[idxs[:last]] - overlap_area)

            need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
            idxs = np.delete(idxs, need_to_be_deleted_idx)

        return output[np.asarray(pick)]

if __name__ == '__main__':
    print('This is the raw detection without Spatial Enhancement, Pls check the Jetson Folder for a full pipeline!')
    m = VariantParkingSlot_Detector()
    path = '/home/dtlam26/Documents/Coral_Project/data/Parking Dataset/Images'
    ll = []
    for p,d,files in os.walk(path):
        if (files):
            for file in files:
                if file.endswith('png'):
                    impath = os.path.join(p,file)
                    o = m(impath,visualize=False,threshold=0.4)
                    ll.append(o[1]['learning_loss'])

    l = np.asarray(ll)
    print('Mean LL', l.mean(), 'Std LL', l.std())
    plt.hist(l,100)
