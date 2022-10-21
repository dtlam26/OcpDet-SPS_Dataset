import os
import json
import cv2
import math
from shapely.geometry import Polygon as P
from meta import ParkingMeta
from loguru import logger


class Filter_Engine:
    def __init__(self):
        self.label_list = []
        self.images = []
        self.meta_path = ''
        self.image_folder = ''

    def load_label_from_folder(self,folder_path):
        self.label_list = [os.path.join(folder_path,p) for p in os.listdir(folder_path) if p.endswith('.json')]
        if not self.label_list:
            logger.info(f"Have not annotated {self.label_list}")

    def load_image_folder(self,folder_path):
        self.image_folder = folder_path
        self.images = [i for i in os.listdir(folder_path) if not i.endswith('.json')]

    def load_meta(self,meta_path):
        self.meta_path = meta_path

    def filter_file_with_rectangle_polygon(self):
        assert self.label_list, "Pls load label folder"
        def check_rectangle(point):
            poly = P(point)
            if math.isclose(poly.minimum_rotated_rectangle.area, poly.area):
                return True
            else:
                return False
        for label in self.label_list:
            found = False

            with open(label,'r') as f:
                annotation = json.load(f)
            for shape in annotation['shapes']:
                if shape['shape_type'] == 'polygon':
                    if check_rectangle(shape['points']):
                        found = True
                        break
            if found:
                os.remove(label)
                logger.debug(label)

    def filter_image_without_label(self,empty_image_data=True,ann_folder_path=''):
        assert self.images and self.label_list, "Pls load image folder and label folder"

        def check_wrong_polygon(s):
            poly = s['points']
            if len(poly)!=4 and s['label']!='mask':
                return True
            return False

        def get_image_name(file_path):
            if '\\' in file_path:
                return file_path.split('\\')[-1]
            else:
                return file_path.split('/')[-1]

        used_images = []
        for label in self.label_list:
            with open(label,'r') as f:
                annotation = json.load(f)

            img_file = get_image_name(annotation['imagePath'])

            if ann_folder_path:
                annotation['imagePath'] = os.path.join(ann_folder_path,img_file)
            else:
                annotation['imagePath'] = img_file

            if annotation['imageData'] and empty_image_data:
                logger.info(f"Redundant img meta  {annotation['imagePath']}")
                annotation['imageData'] = None
            wrong_polygon = False
            for i in range(len(annotation['shapes'])):
                tempt = annotation['shapes'][i]['label'].split('_')
                wrong_polygon |= check_wrong_polygon(annotation['shapes'][i])
                if tempt[0] == '1':
                    annotation['shapes'][i]['label'] = 'available'
                elif tempt[0] == '2':
                    annotation['shapes'][i]['label'] = 'occupied'
                elif tempt[0] == '3':
                    annotation['shapes'][i]['label'] = 'illegal'
                elif tempt[0].lower() == 'vip':
                    annotation['shapes'][i]['label'] = 'available'
                elif tempt[0].lower() == 'ev':
                    annotation['shapes'][i]['label'] = 'available'
                else:
                    annotation['shapes'][i]['label'] = tempt[0]

                if len(tempt) > 1:
                    state = tempt[-1]
                    if state == 'kt':
                        state = 'handicap'
                else:
                    if tempt[0].lower() == 'vip':
                        state = 'vip'
                    elif tempt[0].lower() == 'ev':
                        state = 'ev'
                    else:
                        state = 'normal'

                annotation['shapes'][i]['attr'] = {}
                if 'state' in annotation['shapes'][i]['flags']:
                    annotation['shapes'][i]['attr']['type'] = annotation['shapes'][i]['flags']['state']
                    annotation['shapes'][i]['flags'] = {}
                else:
                    annotation['shapes'][i]['attr']['type'] = state



            if wrong_polygon:
                logger.info(f"Wrong Ann {annotation['imagePath']}")

            used_images.append(img_file)

            with open(label,'w') as f:
                json.dump(annotation,f,indent=4)

        if used_images:
            for r in list(set(self.images) - set(used_images)):
                logger.info(f"Redundant imgs in {self.image_folder}")
                os.remove(os.path.join(self.image_folder,r))

    def filter_uncheck_label_prediction(self):
        assert len(self.images) == len(self.label_list), 'The number of labels vs The number of images are not the same'
        for label in self.label_list:
            with open(label,'r') as f:
                annotation = json.load(f)
            try:
                if annotation['flags']['0']=='__ignore__':
                    os.remove(label)
            except:
                pass

    def filter_meta(self,meta_path,is_test_meta=False):
        meta = ParkingMeta(meta_path)
        remove = 0
        _, coco = meta.convert2COCOformat(inplace=False,return_coco=True)
        for k,v in coco.imgs.items():
            if not v['id'] in coco.imgToAnns:
                logger.debug(f"No Annotation for {v['id']}")
                try:
                    os.remove(coco.imgs[k]["file_name"])
                    remove += 1
                except:
                    pass
                meta.meta["images"].pop(str(k))
        meta.export_meta_as_json(path='meta_complete.json')
        logger.success("DONE! Remove %d"%remove)
        ## TO DO loai cac anh khong gian nhan khoi meta, them attributte building number cho tung video

##Demo Filter

##Train Set
f = Filter_Engine()

#filter both image, and label (DANGEROUS):
f.filter_meta('meta_complete.json')

# meta = ParkingMeta('meta_complete.json')
# _, coco = meta.convert2COCOformat(inplace=False,return_coco=True)

# ann_folder = '/home/dtlam26/Documents/Coral_Project/data/Parking Dataset/Annotation'
# img_folder = '/home/dtlam26/Documents/Coral_Project/data/Parking Dataset/Images'
#
# folders = os.listdir(ann_folder)
# for folder in folders:
#     print(folder)
#     f.load_label_from_folder(os.path.join(ann_folder,folder))
#     f.load_image_folder(os.path.join(img_folder,folder))
#
#     ann_folder_path = os.path.join('../../Images',folder)
#
#     ##Only delete Image according to Meta Info, not Meta modification
#     f.filter_image_without_label(ann_folder_path=ann_folder_path)

    ##legacy
    # f.filter_file_with_rectangle_polygon()
    # f.filter_uncheck_label_prediction()


##Test Set
# f = Filter_Engine()
# ann_folder = '/home/dtlam26/Documents/Coral_Project/data/Parking Dataset/Test'
#
#
# for folder,dir_contents,file_contents in os.walk(ann_folder):
#     if file_contents:
#         f.load_label_from_folder(folder)
#         f.load_image_folder(folder)
#
#         ann_folder_path = os.path.relpath(folder)
#         f.filter_image_without_label(ann_folder_path='')
#         f.filter_file_with_rectangle_polygon()

        # f.filter_uncheck_label_prediction()
