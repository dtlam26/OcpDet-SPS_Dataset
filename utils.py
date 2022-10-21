import os,re
import numpy as np
import cv2
import tensorflow as tf
from shapely.geometry import Polygon as P
import json
import operator
from collections import deque

def image2Polygon(image):
    level_image = image.astype(np.float32)
    level_image = level_image/np.max(level_image)
    image[np.where(image>0)] = 1
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        polygons.append(P(c.reshape(-1,2)))
    return polygons,level_image

def soft_mask_generator(active_anchors,model_input_size=896,target_shape=(1920,1080),threshold=0.6):
    reshape_prediction_tensors = []
    anchor_per_head=[]
    size=[]
    for prediction_tensor in active_anchors:
        if prediction_tensor.shape:
            shape = list(prediction_tensor.shape)
            current_num_anchors = shape[0]*shape[1]*shape[2]
            anchor_per_head.append(shape[0]*shape[1])
            size.append(int(896/shape[0]))
            # print(np.where(prediction_tensor>0.6))
            reshape_prediction_tensors.append(prediction_tensor.reshape(current_num_anchors,1))
    total_anchor = np.concatenate(reshape_prediction_tensors,0)
    mask = np.where(total_anchor>threshold)

    num_anchor_per_loc = 1
    pts = []
    for m in mask[0].tolist():
        for i in range(len(anchor_per_head)):
            if i == 0:
                if m <= anchor_per_head[i]:
                    index = m
                    # max_size = 896/size[i]
                    break
            else:
                if m >= sum(anchor_per_head[:i]) and m <= sum(anchor_per_head[:i+1]):
                    index = m-sum(anchor_per_head[:i])
                    # max_size = 896/size[i]
                    break

        y = index//(model_input_size/size[i]*num_anchor_per_loc)
        x = (index-(model_input_size/size[i])*num_anchor_per_loc*y)//num_anchor_per_loc
        pts.append([((int(x-2)*size[i]),int((y-2)*size[i])),(int((x+2)*size[i]),int((y+2)*size[i]))])

    map = None
    for pt in pts:
        base = np.zeros((model_input_size,model_input_size))
        cv2.rectangle(base, pt[0],pt[1],
                              color=(1,1,1), thickness=-1)
        if map is None:
            map = base
        else:
            map += base

    return image2Polygon(cv2.resize(map.astype(np.uint8),target_shape))

def self_intersection_ratio(A,B):
    return B.intersection(A).area/A.area

def cut_object(img,xy,r):
    r = int(r)
    rectX = int(xy[0] - r)
    rectY = int(xy[1] - r)
    return img[rectY:(rectY+2*r), rectX:(rectX+2*r)], [rectY,rectY+2*r, rectX,rectX+2*r]

def scale(cordinates,format,width,height,normalize=True):
    if normalize:
        f = operator.mul
    else:
        f=operator.truediv
    if type(cordinates)==list:
        cordinates = np.asarray(cordinates)
    if format=='xyxy':
        cordinates[:,:,0] = f(cordinates[:,:,0],width)
        cordinates[:,:,1] = f(cordinates[:,:,1],height)
    elif format=='yxyx':
        cordinates[:,:,0] = f(cordinates[:,:,0],height)
        cordinates[:,:,1] = f(cordinates[:,:,1],width)
    else:
        raise ValueError(format, ' is Not Supported')
    return cordinates

def swap_xy(bboxes):
    bboxes = np.asarray(bboxes)
    new_bboxes = bboxes.copy()
    new_bboxes[:,:4:2] = bboxes[:,1:4:2]
    new_bboxes[:,1:4:2] = bboxes[:,:4:2]
    return new_bboxes

def xy_wh2xmym_wh(bbox):
    new_bbox = []
    new_bbox.append(int(bbox[0]-bbox[2]/2))
    new_bbox.append(int(bbox[1]-bbox[3]/2))
    new_bbox.append(int(bbox[2]))
    new_bbox.append(int(bbox[3]))
    return new_bbox

def xmym_wh2xy_wh(bbox):
    new_bbox = []
    new_bbox.append(int(bbox[0]+bbox[2]/2))
    new_bbox.append(int(bbox[1]+bbox[3]/2))
    new_bbox.append(int(bbox[2]))
    new_bbox.append(int(bbox[3]))
    return new_bbox

def return_rectangle_corners(polygon,format='yxyx',number_cords=4):
    if format=='xyxy':
        y1 = polygon[0]
        x1 = polygon[1]
        y2 = polygon[2]
        x2 = polygon[3]
    elif format=='yxyx':
        y1 = polygon[1]
        x1 = polygon[0]
        y2 = polygon[3]
        x2 = polygon[2]
    else:
        raise ValueError(format, ' is Not Supported')
    p1 = [x1, y1]
    p2 = [x2, y1]
    p3 = [x2, y2]
    p4 = [x1, y2]
    if number_cords==4:
        return [p1, p2, p3, p4]
    elif number_cords==2:
        return [p1,p3]
    else:
        raise ValueError(number_cords, ' can only be 4 or 2')

def strip_abc(string,full_mode=False):
    if full_mode:
        return int(''.join(re.findall(r'\d+', string)))
    return int(''.join(re.findall(r'\d+', string.split('/')[-1])))

def parse_bbox_format(bbox):
    return bbox[0]+bbox[1]


def compute_geometric_overlap_A_to_B(car_points,parked_car_points,iou=False):
    overlaps = np.zeros((len(car_points), len(parked_car_points)))
    for i in range(len(car_points)):
        for j in range(len(parked_car_points)):
            polygon1_shape = P(car_points[i].copy())
            polygon2_shape = P(parked_car_points[j].copy())
            polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
            if iou:
                polygon_union = polygon1_shape.union(polygon2_shape).area
                overlaps[i][j] = polygon_intersection / polygon_union
            else:
                overlaps[i][j] = polygon_intersection/polygon2_shape.area
    return overlaps
