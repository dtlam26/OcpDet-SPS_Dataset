import pandas as pd
import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from meta import ParkingMeta
from utils import strip_abc

class HungarianAllocator(object):
    def __init__(self):
        self._row = 0
        self._col = 0
        self.cost_matrix = None
        self.candidate_row_pos = {}

    def encode_postion(self,candidates):
        assert self.cost_matrix is not None, "Please load_matching first"
        base = 0
        self.candidate_row_pos = {}
        for r in sorted(candidates):
            self.candidate_row_pos[r] = (np.arange(candidates[r])+base).tolist()
            base += candidates[r]


    def load_matching(self,requirements,candidates):
        self._col = max(set(requirements['start_index']))+1
        self._row = sum((list(candidates.values())))
        self.cost_matrix = np.ones((self._row,self._col))*1e9

        self.encode_postion(candidates)
        assert self.candidate_row_pos, "There is no encode info"

        if self._row > self._col:
            self.cost_matrix = np.concatenate([self.cost_matrix,np.zeros((self._row,(self._row-self._col)))],1)
        elif self._col > self._row:
            self.cost_matrix = np.concatenate([self.cost_matrix,np.zeros(((self._col-self._row),self._col))],0)
        else:
            pass
        for _,r in requirements.iterrows():
            if r['park_index'] in self.candidate_row_pos:
                for i in self.candidate_row_pos[r['park_index']]:

                    self.cost_matrix[i][r['start_index']] = r['time_travel_s']


    def perform_assignment(self,return_cost=True):
        row_ind, col_ind = linear_sum_assignment(self.cost_matrix)
        if self._row > self._col:
            select = np.where(col_ind<self._col)
        elif self._col > self._row:
            select = np.where(row_ind<self._row)
        else:
            select = np.arange(self._row)
        row_ind = row_ind[select]
        col_ind = col_ind[select]
        mapping = {}
        count = {}
        assert self.candidate_row_pos, "There is no encode info"

        for c,r in zip(col_ind.tolist(),row_ind.tolist()):
            for p in self.candidate_row_pos:
                if r in self.candidate_row_pos[p]:
                    break
            mapping[c] = p
            if p in count:
                count[p] += 1
            else:
                count[p] = 1

        if return_cost:
            assigned_mat = self.cost_matrix[row_ind, col_ind]
            return mapping, count, assigned_mat[np.where((assigned_mat<1e9)&(assigned_mat>0))].sum()
        else:
            return mapping, count

class AnalyzeResults():
    def __init__(self,result_path='results/results_best.json',test_meta_path='test_meta.json'):
        with open(result_path,'r') as f:
            pred_results = json.load(f)
        parking = ParkingMeta('test_meta.json',is_test_meta=True)
        _, coco = parking.convert2COCOformat(inplace=False,return_coco=True)
        self.info = coco.loadRes(pred_results['mask_results'])
        categories = self.info.cats
        for c in categories:
            if categories[c]['name'] == 'available':
                self.vacant_key = categories[c]['id']
                break

    def get_image_vacant_slots(self,path):
        id =  strip_abc(path,True)
        dets = self.info.imgToAnns[int(id)]
        vacant = 0
        for d in dets:
            if d['category_id'] == self.vacant_key:
                vacant += 1
        return vacant

def error_cal(gt,det):
    cost_err = abs(gt['budget']-det['budget'])/(1.5*gt['budget'])
    assignment_error = []
    for i in range(6):
        if i in gt['count']:
            det_res = 0 if i not in det['count'] else det['count'][i]
            assignment_error.append(abs(gt['count'][i] - det_res)/sum(list(gt['count'].values())))
        elif i in det['count']:
            assignment_error.append(1)
    return sum(assignment_error)/len(assignment_error),cost_err

time = {'3pm':15,'4pm':16,'5pm':17,'6pm':18}

with open('road_meta.json','r') as f:
    config = json.load(f)

hg = HungarianAllocator()

target = config['parking_locations']
#read from traffic data
global_traffic_data = pd.read_csv('records.csv')
days = set(global_traffic_data['day'])



ar = AnalyzeResults()

gts = {}
dets = {}
total_error = {}


for i,d in enumerate(days,start=1):
    j = i%5
    if j == 0:
        j = 5
    selected_day = f'day_{j}'
    records_per_day = global_traffic_data[global_traffic_data['day']==d]
    if selected_day not in total_error:
        total_error[selected_day] = {'cost_error':[],'assignment_error':[]}
    gts[selected_day] = {}
    dets[selected_day] = {}
    for t in time:
        requirements = records_per_day[records_per_day['record_hour']==time[t]]
        candidates = {}
        predict_candidates = {}
        for f,_,_ in os.walk('Test'):
            if f.endswith(os.path.join(selected_day,t)):
                # print(f)
                stages = f.split('/')
                park_location_index = int(stages[-3].split('_')[-1]) -1
                for content in os.listdir(f):
                    if content.endswith(".json"):
                        predict_candidates[park_location_index] = ar.get_image_vacant_slots(os.path.join(f,content))
                        with open(os.path.join(f,content),'r') as _json:
                            ann = json.load(_json)
                        candidates[park_location_index] = \
                            sum([1 if s['label']=='available' else 0 for s in ann['shapes']])
            else:
                continue
        if candidates:
            hg.load_matching(requirements,candidates)
            map_gt,count,cost = hg.perform_assignment()
            gts[selected_day][t] = {'map': map_gt, 'count': count, 'budget':cost}

            hg.load_matching(requirements,predict_candidates)
            map_dt,count,cost = hg.perform_assignment()
            dets[selected_day][t] = {'map': map_dt,'count': count,'budget':cost}


            assignment_error, cost_error = error_cal(gts[selected_day][t],dets[selected_day][t])
            total_error[selected_day]['assignment_error'].append(assignment_error)
            total_error[selected_day]['cost_error'].append(cost_error)

print(total_error)
