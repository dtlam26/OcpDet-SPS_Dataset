# import tensorflow.compat.v1 as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)
import json
import pandas as pd
import os
from pathlib import Path
from shapely.geometry import Polygon as P
from shapely.geometry import box as B

from utils import *
import contextlib2
from tqdm import tqdm
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import VariantParkingSlot_Detector
from object_detection.utils import dataset_util
import matplotlib.pyplot as plt
from object_detection.dataset_tools import tf_record_creation_util
import sys
import tensorflow.compat.v1 as tf1


class ParkingMeta():
    def __init__(self,meta_path='',bb_df='',is_test_meta=False,keep_format=True):
        self.obj_categories = {1:'occupied',2:'available',3:'illegal',4:'restricted'}

        default_categories = [{
            "supercategory": "parking",
            "id": k,
            "name": v,
            "keypoints": [
                'top_left',
                'top_right',
                'bot_right',
                'bot_left'
            ]
        } for k,v in self.obj_categories.items()]
        self.writer = None
        self.shard = False
        self.override_annotations = False
        self.coco = False
        self.index = 0
        self.online = False
        self.is_test_meta = is_test_meta
        if is_test_meta:
            if os.path.isdir(meta_path):
                self.meta = {'videos':{},'images':{},'annotations':[],'categories':default_categories}
                self.load_anns_from_test_folder(meta_path)
                self.export_meta_as_json('test.json')
            else:
                assert meta_path.endswith('json'), 'Only accept TestJson or A Test Foler path'
                with open(meta_path,'r') as f:
                    self.meta = json.load(f)
                logger.info("Create backup")
                self.export_meta_as_json(meta_path.replace('.json','_backup.json'))
        else:
            if os.path.isfile(meta_path):
                with open(meta_path,'r') as f:
                    self.meta = json.load(f)
                logger.info("Create backup")
                self.export_meta_as_json(meta_path.replace('.json','_backup.json'))
            else:
                self.meta = {'videos':{},'images':{},'annotations':[],'categories':default_categories}

        if self._check_coco_format() and not keep_format:
            self.revert_from_COCOformat(inplace=True)

        if bb_df:
            self.bb_df = pd.read_csv(bb_df)
        else:
            self.bb_df = pd.DataFrame(columns= ['image_path','obj_class','x1','y1','x2','y2'])

        logger.info(f"Total imgs: {len(self.meta['images'])} | Total labels: {len(self.meta['annotations'])}")
        if len(self.meta['annotations']):
            self.annotation_summary()

    def annotation_summary(self):
        categories = {cat['id']:[cat['name'],0] for cat in self.meta['categories']}
        for ann in self.meta['annotations']:
            categories[ann['category_id']][-1] += 1
        for c in list(categories.values()):
            logger.info(f"{c[0]}: {c[1]} anns")

    def load_anns_from_test_folder(self,test_folder):
        park_loc = [
            [37.453852,126.950869],
            [37.455393,126.954787],
            [37.454972,126.951899],
            [37.460319,126.948958],
            [37.463096,126.954647],
            [37.457313,126.952549]
        ]
        for folder,dir_contents,file_contents in os.walk(test_folder):
            if file_contents:
                label_list = [os.path.join(folder,p) for p in os.listdir(folder) if p.endswith('.json')]
                park = park_loc[strip_abc(folder.split("/")[-3])-1]
                for label in label_list:
                    self.meta['images'][str(strip_abc(label,True))] = {
                        "id": strip_abc(label,True),
                        "location": park
                    }
                self.export_bbs_folder2tfrecord(label_list)

    def _check_coco_format(self):
        if type(self.meta['images'])==list:
            logger.info("COCO format")
            self.coco = True
            return True
        else:
            return False

    def set_override_annotations(self):
        self.override_annotations = True
        self.meta['annotations'] = []

    def set_obj_categories(self,cat):
        assert type(cat)==dict, "Please provide categories in dict format {1: cat, 2: cat ....}"
        self.obj_categories = cat
        if self.override_annotations:
            self.meta['categories'] = [{
                "supercategory": "parking",
                "id": k,
                "name": v,
                "keypoints": [
                    'top_left',
                    'top_right',
                    'bot_right',
                    'bot_left'
                ]
            } for k,v in self.obj_categories.items()]

    def add_meta(self, tag, meta_obj):
        assert meta_obj.hasattr('id'), "meta_object must contain id attribute"
        self.meta[tag].update({str(meta_obj.id): meta_obj.__dict__})

    def export_meta_as_json(self,path='meta.json'):
        if self.coco:
            self.convert2COCOformat()

        with open(path,'w') as f:
            json.dump(self.meta,f,indent=4, sort_keys=True)

    def export_meta_as_csv(self,path='bbs.csv'):
        assert not self.bb_df.empty, "Can't export empty DataFrame"
        self.bb_df.to_csv(path)

    def export_bbsTocsv_from_folder(self,folder):
        """
        | image_path | image_class (optional) | obj_class | x1 | y1 | x2 | y2 |
        |------------|------------------------|-----------|----|----|----|----|
        |            |                        |           |    |    |    |    |
        """
        for ann_file in os.listdir(folder):
            with open(os.path.join(folder,ann_file),'r') as f:
                ann = json.load(f)
            selected_shapes = ann['shapes']
            img_name = ann['imagePath'].split('/')[-1]
            img_id = strip_abc(img_name)
            img_path = os.path.abspath(self.meta['images'][str(img_id)]['img_path'])
            for s in selected_shapes:
                # try:
                xyxy = return_rectangle_corners(P(s['points']).bounds,format='yxyx',number_cords=2) #yxyx
                # except:
                #     print(img_name,s['points'])
                x1,y1 = xyxy[0]
                x2,y2 = xyxy[1]
                obj_class = s['label']
                self.bb_df = self.bb_df.append(dict(x1=x1,x2=x2,y1=y1,y2=y2,
                                obj_class=obj_class,image_path=img_path),ignore_index=True)

    def export_bbs_folder2tfrecord(self,list_or_folder,image_format = b'jpg'):
        if (self.writer is None and not self.shard) or (not (self.writer and self.shard)) or self.is_test_meta:
            skip_tf_records = True
            logger.warning("You wont have any tfrecords as there is none specify, only update meta annotation")
        else:
            skip_tf_records = False

        if type(list_or_folder)==list:
            list_file = list_or_folder
        elif os.path.isdir(list_or_folder):
            list_file = [os.path.join(folder,p) for p in os.listdir(folder)]
        else:
            raise ValueError('Only accep: a list of anns path or a folder containing anns')

        counter = tqdm(total=len(list_file))

        for index, ann_file in enumerate(list_file):
            self.index+=index
            with open(ann_file,'r') as f:
                ann = json.load(f)
            selected_shapes = ann['shapes']
            if self.is_test_meta:
                img_attr = ann['imagePath'].split('/')[-1].split('.')
                img_name = img_attr[0]
                img_path = ann_file.replace('json',img_attr[-1])
                img_id = str(strip_abc(img_path,True))
                self.meta['images'][img_id]['file_name'] = img_path
            else:
                img_name = ann['imagePath'].split('/')[-1]
                img_id = str(strip_abc(img_name))
                img_path = os.path.abspath(self.meta['images'][str(img_id)]['file_name'])


            if self.online:
                encoded_jpg = requests.get(str(img_path)).content
            else:
                if 'http' in img_path:
                    raise ValueError("Please turn on online mode")
                with tf1.gfile.GFile(img_path, 'rb') as fid:
                    encoded_jpg = fid.read()
            _ = tf1.image.decode_image(encoded_jpg)

            height,width,channel = _.shape

            if not skip_tf_records:
                img = cv2.imread(img_path)

                img_path = img_path.encode('utf8')

                feature={
                    'image/height': dataset_util.int64_feature(height),
                    'image/width': dataset_util.int64_feature(width),
                    'image/filename': dataset_util.bytes_feature(img_path),
                    'image/source_id': dataset_util.bytes_feature(img_id.encode('utf8')),
                    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                    'image/format': dataset_util.bytes_feature(image_format),
                }

            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            points = []
            x = []
            y = []

            for det_i,s in enumerate(selected_shapes):
                if s['shape_type'] != 'polygon':
                    continue

                poly = s['points']

                if s['label'] == 'mask':
                    self.meta['images'][img_id]['mask'] = np.asarray(poly).tolist()
                    continue
                elif len(poly)!=4:
                    raise ValueError(f"Wrong annotation, pls firstly filter: {img_name}")
                    continue

                polygon = P(poly)
                xyxy = return_rectangle_corners(polygon.bounds,format='yxyx',number_cords=2) #yxyx

                poly = np.asarray(poly)
                x.extend((poly[:,0]/width).tolist())
                y.extend((poly[:,1]/height).tolist())
                x1 = xyxy[0][0]
                x2 = xyxy[1][0]
                y1 = xyxy[0][1]
                y2 = xyxy[1][1]
                xmins.append(x1/width)
                ymins.append(y1/height)
                xmaxs.append(x2/width)
                ymaxs.append(y2/height)
                classes_text.append(s['label'].encode('utf-8'))
                cls_id = list(self.obj_categories.keys())[list(self.obj_categories.values()).index(s['label'])]
                classes.append(cls_id)
                area = polygon.area

                #bbox xyxy => xywh (xmym_wh will match the COCO format)
                self.meta['annotations'].append({
                    'segmentation': [],
                    'iscrowd': 0,
                    'num_keypoints': 4,
                    'keypoints': np.concatenate([poly,np.ones((len(poly),1))*2],-1).reshape(1,-1).tolist(),
                    'area': area,
                    'image_id': int(img_id),
                    'bbox': [int((x1+x2)/2),int((y1+y2)/2),int(x2-x1),int(y2-y1)],
                    'category_id': cls_id,
                    'id': int(img_id+str(det_i))
                })

            if not skip_tf_records:
                feature.update({
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                'image/object/keypoint/x': dataset_util.float_list_feature(x),
                'image/object/keypoint/y': dataset_util.float_list_feature(y),
                'image/object/keypoint/visibility': dataset_util.int64_list_feature([2]*len(y)),
                'image/object/keypoint/text': dataset_util.bytes_list_feature([b'top_left',b'top_right',b'bot_right',b'bot_left']*int(len(y)/4)),
                })

            # for p in zip(x,y):
            #     cv2.circle(img,(int(p[0]*width),int(p[1]*height)),5,(255,122,22),2)

            # for x1,y1,x2,y2 in zip(xmins,ymins,xmaxs,ymaxs):
            #     cv2.rectangle(img,(int(x1*width),int(y1*height)),(int(x2*width),int(y2*height)),(32,122,22),2)

            # plt.figure(figsize=(15,15))
            # plt.imshow(img)

                tf_example = tf1.train.Example(features=tf1.train.Features(feature=feature))
                if self.shard:
                    output_shard_index = self.index % self.num_shards
                    self.writer[output_shard_index].write(tf_example.SerializeToString())
                else:
                    self.writer.write(tf_example.SerializeToString())
            counter.update(1)

    def create_tf_writer(self,output_folder,file,obj_categories,shard=True,num_shards=10,online=False,override_annotations=True):
        assert not self.is_test_meta, "Can't create Tf Records for Test Set"
        if override_annotations:
            self.set_override_annotations()

        self.shard=shard
        self.num_shards=num_shards
        self.online=online

        if obj_categories:
            self.set_obj_categories(obj_categories)

        self.index=0
        if not os.path.isdir(os.path.join('tfrecords',output_folder)):
            os.makedirs(os.path.join('tfrecords',output_folder))

        msg = ''
        for idx, name in obj_categories.items():
            msg = msg + "item {\n"
            msg = msg + " id: " + str(idx) + "\n"
            msg = msg + " name: '" + name + "'\n}\n\n"
        with open(os.path.join(os.getcwd(),'tfrecords',output_folder,'obj_map.pbtxt'), 'w') as f:
            f.write(msg[:-1])
            f.close()
        self.output_path = os.path.join('tfrecords',output_folder,file)

        if self.shard:
            print('--sharding--')
            self.output_path = self.output_path+'.record'
            self.tf_record_close_stack =  contextlib2.ExitStack()
            self.writer = tf_record_creation_util.open_sharded_output_tfrecords(
              self.tf_record_close_stack, self.output_path, num_shards = self.num_shards)
        else:
            self.output_path = self.output_path+'.tfrecord'
            self.writer = tf1.python_io.TFRecordWriter(self.output_path)

        self.output_path = os.path.join(os.getcwd(), self.output_path)

    def close_tf_writer(self):
        assert not self.is_test_meta, "Can't Close Tf Records for Test Set"
        if self.writer is not None:
            if not self.shard:
                self.writer.close()
            else:
                for w in self.writer:
                    w.close()
                self.tf_record_close_stack.close()
            print('Successfully created the TFRecords: {}'.format(self.output_path))

    def convert2COCOformat(self,inplace=True,return_coco=False):
        assert not self.coco, "Already COCO format"
        images = [v for k,v in self.meta['images'].items()]
        anns = []
        for ann in self.meta['annotations']:
            ann['bbox'] = xy_wh2xmym_wh(ann['bbox'])
            anns.append(ann)

        if inplace:
            self.meta['images'] = images
            self.meta['annotations'] = anns
        else:
            meta = self.meta.copy()
            meta['images'] = images
            meta['anns'] = anns

        if return_coco:
            coco = COCO()
            """anns : xmin ymin width height"""
            if inplace:
                coco.dataset = self.meta
            else:
                coco.dataset = meta
            coco.createIndex()
            return images, coco
        else:
            return images

    def revert_from_COCOformat(self):
        self.meta['images'] = {str(v['id']):v for v in self.meta['images']}
        anns = []
        for ann in self.meta['annotations']:
            ann['bbox'] = xmym_wh2xy_wh(ann['bbox'])
            anns.append(ann)
        self.meta['annotations'] = anns

    def benchmark(self,model,mask_on=False,threshold=0.4,iouThrs=[],override=False,store_path='results/results.json',
        filter_on=[],label_id_offset=1,selected_ids=[]):
        assert self.is_test_meta, "Benchmark can only perform with test set"
        assert isinstance(model,VariantParkingSlot_Detector), "Please provide model by VariantParkingSlot_Detector"

        if mask_on:
            masked_meta = self.meta
            anns = {}

        if not self._check_coco_format():
            logger.info("Convert COCO format")
            _, coco = self.convert2COCOformat(inplace=False,return_coco=True)
        else:
            coco = COCO()
            coco.dataset = self.meta
            coco.createIndex()

        if os.path.isfile(store_path) and not override:
            with open(store_path,'r') as f:
                results = json.load(f)
            if mask_on:
                for ann in masked_meta['annotations']:
                    if str(ann['image_id']) not in anns:
                        anns[str(ann['image_id'])] = []
                    anns[str(ann['image_id'])].append(ann)

                masked_meta['annotations'] = anns

                for img_id in tqdm(coco.imgToAnns,desc='Mask Anns...',colour='yellow'):
                    img_meta = coco.imgs[img_id]

                    img = img_meta['file_name']
                    img = cv2.imread(img)

                    if 'mask' in img_meta:
                        # init_mask = np.zeros(img.shape[:-1])
                        # mask = cv2.fillPoly(init_mask, pts=[np.asarray(img_meta['mask']).astype(np.int32)],color=(1,1,1))
                        mask = P(img_meta['mask'])
                        new_anns = []

                        for ann in masked_meta['annotations'][str(img_id)]:
                            # ann_center = (int(ann['bbox'][0]),int(ann['bbox'][1]))
                            # try:
                            #     if mask[ann_center[1],ann_center[0]]==1:
                            #         new_anns.append(ann)

                            ann_box = B(ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3])

                            try:
                                if self_intersection_ratio(ann_box,mask) > 0.5:
                                    new_anns.append(ann)
                            except:
                                logger.error(f"Annotation is out of bound at {masked_meta['images'][str(img_id)]}")

                        masked_meta['annotations'][str(img_id)] = new_anns
        else:
            results = {"results":[]}
            if mask_on:
                results.update({"mask_results":[]})
                masked_meta = self.meta.copy()
                anns = {id:[] for id in masked_meta['images']}
                for ann in masked_meta['annotations']:
                    anns[str(ann['image_id'])].append(ann)
                masked_meta['annotations'] = anns

            for img_id in tqdm(coco.imgToAnns,desc='Extracting Results...',colour='green'):
                img_meta = coco.imgs[img_id]

                img = img_meta['file_name']
                img = cv2.imread(img)

                if 'mask' in img_meta:
                    # init_mask = np.zeros(img.shape[:-1])
                    # mask = cv2.fillPoly(init_mask, pts=[np.asarray(img_meta['mask']).astype(np.int32)],color=(1,1,1))
                    mask = P(img_meta['mask'])
                else:
                    mask = None
                h,w,_ = img.shape
                output, additional_output = model(img,threshold=threshold)
                soft_mask = []
                if additional_output:
                    for k in additional_output:
                        if k not in results:
                            results.update({k:{}})
                            if k == 'active_anchors':
                                results.update({'active_anchors_results':[]})
                        else:
                            if k == 'learning_loss':
                                results[k][img_id]=additional_output[k]
                            else:
                                soft_mask = soft_mask_generator(additional_output[k],threshold=0.4,target_shape=(w,h))
                                results["active_anchors"][img_id] = []
                for i in range(len(output)):
                    o = output[i]
                    category_id = int(o[4]) + label_id_offset
                    confidence = float(o[-1])
                    x1 = o[1]
                    y1 = o[0]
                    x2 = o[3]
                    y2 = o[2]


                    bbox = [int(x1),int(y1),int(x2-x1),int(y2-y1)]
                    box = B(int(x1),int(y1),int(x2),int(y2))
                    results["results"].append({
                        "image_id": img_id, "category_id": category_id, "bbox": bbox, "score": confidence
                    })
                    if soft_mask:
                        match_ratio = []
                        for sm in soft_mask[0]:
                            match_ratio.append(self_intersection_ratio(box,sm))
                        match_ratio = max(match_ratio)
                        heat_box = float(np.max(soft_mask[1][int(y1):int(y2),int(x1):int(x2)]))
                        results["active_anchors"][img_id].append([match_ratio,heat_box])
                        if match_ratio*confidence>0.4:
                            # if confidence > 0.8:
                            #     if confidence*heat_box<0.2:
                            #         continue
                            #     else:
                            #         confidence = confidence*(1+heat_box)
                            results["active_anchors_results"].append({
                                "image_id": img_id, "category_id": category_id, "bbox": bbox, "score": confidence
                            })

                    if mask_on:
                        # center = (int((x1+x2)/2),int((y1+y2)/2))
                        if mask is None:
                            results["mask_results"].append({
                                "image_id": img_id, "category_id": category_id, "bbox": bbox, "score": confidence
                            })
                        else:
                            # if mask[center[1],center[0]]==1:
                            if self_intersection_ratio(box,mask) > 0.5:
                                results["mask_results"].append({
                                    "image_id": img_id, "category_id": category_id, "bbox": bbox, "score": confidence
                                })
                if mask_on:
                    if mask is not None:
                        new_anns = []
                        for ann in masked_meta['annotations'][str(img_id)]:
                            ann_box = B(ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3])

                            try:
                                if self_intersection_ratio(ann_box,mask) > 0.5:
                                    new_anns.append(ann)
                            except:
                                logger.error(f"Annotation is out of bound at {masked_meta['images'][str(img_id)]}")

                        masked_meta['annotations'][str(img_id)] = new_anns

            with open(store_path,'w') as f:
                json.dump(results,f,indent=4, sort_keys=True)

        if mask_on:
            gt_mask = COCO()
            list_anns = []
            for v in masked_meta['annotations'].values():
                list_anns.extend(v)
            # for v in list_anns:
            #     v['bbox'] = xy_wh2xmym_wh(v['bbox'])
            masked_meta['annotations'] = list_anns
            masked_meta['images'] = [v for k,v in masked_meta['images'].items()]
            gt_mask.dataset = masked_meta
            gt_mask.createIndex()

        errors = {}
        for k in results:
            errors[k] = []
            if 'anchor' in k or 'learning_loss' in k:
                continue


            gt = coco
            if k == 'mask_results':
                if mask_on:
                    gt = gt_mask
                else:
                    continue
            logger.info(f"Benchmark on {k}: {len(results[k])}")
            dt = gt.loadRes(results[k])

            if filter_on:
                ll_info = results['learning_loss']
                ll_thresh = filter_on[0]
                if len(filter_on)>1:
                    aa_info = gt.loadRes(results["active_anchors_results"])
                    aa = results['active_anchors']

                else:
                    aa_info = None
                selected_ids = []
                rank_error = {}
                for img_id in ll_info:
                    if aa_info is not None:
                        spatial_ratio = np.asarray(aa[img_id]).reshape(-1,2)[:,0]
                        heat_ratio = np.asarray(aa[img_id]).reshape(-1,2)[:,1]
                        # spatial_ratio = spatial_ratio[:,0]*spatial_ratio[:,1]
                        spatial_ratio = spatial_ratio[np.where(spatial_ratio<0.8)]

                        if spatial_ratio.shape[0]:
                            spatio_ratio_error = 1 - spatial_ratio.mean()*heat_ratio.min()
                        else:
                            # spatio_ratio_error = 1
                            try:
                                spatio_ratio_error = 1 - heat_ratio.min()
                            except:
                                spatial_ratio_error = 1
                        if ll_info[img_id] < ll_thresh:
                            ll_error = 0
                        else:
                            ll_error = (ll_info[img_id] - ll_thresh)/ll_thresh
                        errors[k].append([spatio_ratio_error,ll_error,img_id])
                        error = ll_error*filter_on[1] + spatio_ratio_error*(1-filter_on[1])
                    else:
                        error = ll_info[img_id]

                    rank_error[int(img_id)] = error
                non_zero = {k:v for k,v in rank_error.items() if v>0}
                zeros = [k for k,v in rank_error.items() if v==0]
                zeros.extend([k for k, v in sorted(non_zero.items(), key=lambda item: item[1])][:-100])
                dt = aa_info
                selected_ids=zeros
                # selected_ids=list(aa_info.imgs.keys())
            evaluation = COCOeval(gt,dt,'bbox')
            if selected_ids:
                logger.debug(f"Using: {len(selected_ids)} images")
                evaluation.params.imgIds = selected_ids
            if iouThrs:
                evaluation.params.iouThrs = iouThrs
            # evaluation.params.areaRng = [[0,0.15e5],[0.15e5,1e5],[1e5,100e5]]
            # evaluation.params.areaRngLbl = ['small', 'medium', 'large']
            evaluation.params.areaRng = [[0.15e5,100e5],[0.15e5,1e5],[1e5,100e5]]
            evaluation.params.areaRngLbl = ['all','medium', 'large']
            evaluation.params.catIds = [1,2]
            # evaluation.params.maxDets = [1, 10, 20]
            evaluation.evaluate()
            logger.info("Full Evaluation")
            evaluation.accumulate()
            evaluation.summarize()

            for c in gt.cats:
                cat = gt.cats[c]
                logger.info(f"Benchmark per class: {cat['name']}")
                evaluation.params.catIds = [cat['id']]
                evaluation.evaluate()
                evaluation.accumulate()
                evaluation.summarize()
        return errors

class Meta():
    def __init__(self,**kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def add(self,**kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def hasattr(self,attr):
        try:
            self.__getattribute__(attr)
            return True
        except:
            return False

# parking = ParkingMeta('test_meta.json')
#
# _, coco = parking.convert2COCOformat(inplace=True,return_coco=True)
#
# parking.export_meta_as_json("test_meta_coco.json")

if __name__ == '__main__':
    # mode = sys.argv[1]
    mode = 'train'
    if mode:
        logger.info("MODE: %s"%mode)
        if mode == 'train':
            parking = ParkingMeta('meta.json')
            anns_list = os.listdir('Annotation')

            # Override for new categories
            # parking.set_override_annotations()
            # parking.set_obj_categories({1:'occupied',2:'available',3:'illegal',4:'restricted'})

            for folder in tqdm(anns_list,colour="green",desc="Main Progress"):
                # parking.add2bbs_csv_from_folder(os.path.join('Annotation',folder))
                # parking.export_bbs2dataframe()
                parking.export_bbs_folder2tfrecord(os.path.join('Annotation',folder))
                # parking.index = parking.index+1

            parking.close_tf_writer()

        elif mode=='test':
            parking = ParkingMeta('Test',is_test_meta=True)

        elif mode =='benchmark':
            parking = ParkingMeta('test_meta.json',is_test_meta=True)
            model = VariantParkingSlot_Detector('./weights/mask_mul/saved_model')
            # ids = []
            # for f,_,_ in os.walk('Test'):
            #     if 'day_5' in f:
            #         for content in os.listdir(f):
            #             if content.endswith(".json"):
            #                 ids.append(strip_abc(os.path.join(f,content),True))
            errors = parking.benchmark(model,mask_on=False,override=True,threshold=0.2,filter_on=[],
                                # selected_ids=ids,
                                store_path='results/results_mul.json')

        print('Complete')
    else:
        print('No mode is Selected')

# with open('results/results_best_new.json','r') as f:
#     results = json.load(f)
#
# aas = []
# for aa in list(results['active_anchors'].values()):
#     aas.extend(aa)
# aas = np.asarray(aas).reshape(-1,2)[:,1].reshape([-1]).tolist()
#
# a = plt.hist([e[0] for e in errors['results'] if e[0]>0],100)
# a = plt.hist([e for e in aas],200)
# for e in errors['results']:
#     if e[1] >0.4:
#         print(e)
