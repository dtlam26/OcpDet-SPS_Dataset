import numpy as np
import time
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
from pathlib import Path

COLORS = (cm.get_cmap('Dark2', 8).colors[:,:3]*255).astype(np.int32).tolist()

class Pointwise(object):
    def __init__(self,x,y=None):
        if y:
            self.xy = (x,y)
        else:
            self.xy = x
        self.xy = (int(self.xy[0]),int(self.xy[1]))
    def _check_tuple(self,p):
        if type(p)==tuple:
            return Pointwise(p)
        return p

    def __add__(self,p):
        p = self._check_tuple(p)
        if type(p)==Pointwise:
            return Pointwise((self.xy[0] + p.xy[0], self.xy[1] + p.xy[1]))
        else:
            return Pointwise((self.xy[0] + p, self.xy[1] + p))
    def __sub__(self,p):
        p = self._check_tuple(p)
        if type(p)==Pointwise:
            return Pointwise((self.xy[0] - p.xy[0], self.xy[1] - p.xy[1]))
        else:
            return Pointwise((self.xy[0] - p, self.xy[1] - p))
    def __mul__(self,p):
        p = self._check_tuple(p)
        if type(p)==Pointwise:
            return Pointwise((self.xy[0] * p.xy[0], self.xy[1] * p.xy[1]))
        else:
            return Pointwise((self.xy[0] * p, self.xy[1] * p))
    def __truediv__(self,p):
        p = self._check_tuple(p)
        if type(p)==Pointwise:
            return Pointwise((self.xy[0] / p.xy[0], self.xy[1] / p.xy[1]))
        else:
            return Pointwise((self.xy[0] / p, self.xy[1] / p))


class Point(object):
    def __init__(self,width=1,height=1):
        self.width = width
        self.height = height

    def __call__(self,x,y=None):
        if y:
            return Pointwise(x*self.width,y*self.height)
        else:
            return Pointwise(x[0]*self.width,x[1]*self.height)


class Property():
    def __init__(self,normalize,mean=[],scale=[],threshold=0.2,
                 additional_tensor_index=None,architecture='ssd'):
        self.normalize = 2,
        if normalize == 3:
            assert mean and scale, 'Custom normalization requires different mean and scale'
        self.mean = mean
        self.scale = scale
        self.threshold = threshold
        self.additional_tensor_index = additional_tensor_index
        self.architecture = architecture

def zero_pad_img(image,tblr=[0,0,0,0]):
    pad_0 = [0,0,0]
    return cv2.copyMakeBorder(image.copy(),tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=pad_0)

def rename(path, new_name):
    suffix = path.suffix
    return path.with_name(f'{new_name}.t').with_suffix(suffix)

def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def swap_xy(x):
    y = np.copy(x)
    for i in range(x.shape[-1]):
        if i%2:
            y[:,i] = x[:,i-1]
        else:
            y[:,i] = x[:,i+1]
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, multi_label=False,max_det=300,
                       width=1920,height=1080,object_score=True,bb_format_code=1):
    """Runs Non-Maximum Suppression (NMS) on inference results
         prediction: as an array
         if             object_score [n_anchors x [bboxes[4]]+[object_score[1]]+[multilass_score[num_class]]
         else             object_score [n_anchors x [bboxes[4]]+[multilass_score_with_background[num_class+1]]
         bb_format_code: 0 - center width height  1 - 4 corners
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Settings
    min_wh, max_wh = 0.001, 1  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    nc = prediction.shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    indexes = np.arange(len(prediction))

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    t = time.time()
    output = np.zeros((0, 6))

    # Compute conf
    if object_score:
        prediction[:, 5:] *= prediction[:, 4:5]  # conf = obj_conf * cls_conf
        multi_score_index = 5
        xc = np.where(prediction[:, 4] > conf_thres)  # candidates
        prediction = prediction[xc]  # confidence
        indexes = indexes[xc]
    else:
        multi_score_index = 4

    if bb_format_code==0:
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        whc = np.where((prediction[:, 2:4] > min_wh) & (prediction[:, 2:4] < max_wh)) # width-height
        indexes = indexes[whc[0]]
        prediction = prediction[whc[0]]
        box = xywh2xyxy(prediction[:, :4])


    elif bb_format_code==1:
        box = prediction[:, :4]
    else:
        raise ValueError(f'Not support this bounding box format code: {bb_format_code}')

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = np.where(prediction[:, multi_score_index:] > conf_thres)

        if object_score:
            indexes = indexes[i]
            xxyxy, conf, cls = box[i], prediction[i, j + multi_score_index].reshape(-1,1), j.astype(np.float32).reshape(-1,1)
        else:
            c = np.where(j>0)
            indexes = indexes[i[c]]
            xxyxy, conf, cls = box[i[c]], prediction[i[c], j[c] + multi_score_index].reshape(-1,1), j[c].astype(np.float32).reshape(-1,1)

    else:  # best class only
        j = np.argmax(prediction[:, multi_score_index:],1)
        max_conf = np.max(prediction[:,multi_score_index:],1)
        if object_score:
            i = np.where(max_conf > conf_thres)
        else:
            i = np.where((max_conf > conf_thres) & (j>0))
        indexes = indexes[i]
        xyxy = box[i]
        cls = j[i].reshape(-1,1).astype(np.float32)
        conf = max_conf[i].reshape(-1,1)


    # Check shape
    n = xyxy.shape[0]  # number of boxes

    if not n:  # no boxes
        return output, []
    else:
        print(xyxy.shape,cls.shape,cls.shape)
        x = np.concatenate([xyxy,conf,cls],1)
        x = x[np.argsort(x[:, 4])][:max_nms,:]  # sort by confidence
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = single_class_non_max_suppression(boxes, scores, iou_thresh=iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#             iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#             weights = iou * scores[None]  # box weights
#             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#             if redundant:
#                 i = i[iou.sum(1) > 1]  # require redundancy

        output = x[i]
        indexes = indexes[i]
        print(time.time()-t)
        return output, indexes.tolist()

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.25, iou_thresh=0.45, keep_top_k=-1,skip_sort=True):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []
    if skip_sort:
        idxs = np.arange(len(confidences))
    else:
        conf_keep_idx = np.where(confidences > conf_thresh)[0]
        bboxes = bboxes[conf_keep_idx]
        confidences = confidences[conf_keep_idx]
        idxs = np.argsort(confidences)


    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)


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
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return np.asarray(pick)

def draw_location(frame,output,label_map={},width=1,height=1,shape='rect'):
    """output order: xyxy,conf,cls_index"""
    p = Point(width,height)
    assert type(output)==zip or type(output)==list or type(output)==np.ndarray, 'None supported format'
    for o in output:
        c = int(o[-1])%len(COLORS)
        center = (p(o[:2])+p(o[2:]))/2
        if label_map:
            if int(o[-1]) in label_map:
                label = '%s: [%.3f]'%(label_map[int(o[-1])],o[-2])
                labelSize,_ =cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,6)
                cv2.rectangle(frame,(center-(labelSize[0]/(2*width),labelSize[1]*1.5/(2*height))).xy,
                              (center+(labelSize[0]/(2*width),labelSize[1]*1.5/(2*height))).xy,color=COLORS[c],thickness=-1)
                cv2.putText(frame,label, (center-(labelSize[0]/(2*width),-labelSize[1]/(2*height))).xy,cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5,color = (255, 255, 255),thickness=2)

                if shape == 'rect':
                    cv2.rectangle(frame,p(o[:2]).xy,p(o[2:]).xy,color=COLORS[c],thickness=3)
                else:
                    cv2.circle(frame,center.xy,int((o[2]-o[0])*width/3),COLORS[c],2)
        else:
            if shape == 'rect':
                cv2.rectangle(frame,p(o[:2]).xy,p(o[2:]).xy,color=COLORS[c],thickness=3)
            else:
                cv2.circle(frame,center.xy,int((o[2]-o[0])*width/3),COLORS[c],2)


def show(frame):
    plt.figure(figsize=(15,15))
    plt.imshow(frame)
