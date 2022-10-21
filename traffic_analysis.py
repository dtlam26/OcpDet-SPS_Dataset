import json
import pandas as pd
import datetime
import os
import random
from loguru import logger

"""
    grade: 01: smooth, 02: slow, 03: delay
"""

def analyze_flow(traffic_flow):
    previous_node = {}
    flow = 0
    for node in traffic_flow:
        if node:
            if previous_node:
                if node['fromnodeid'] == previous_node['fromnodeid']:
                    continue
            else:
                previous_node = node
            # ttime = node['ttime']
            ttime = node['len']/(node['speed']/3.6)
            # flow += float(node['ttime'])
        else:
            if previous_node:
                if int(previous_node['grade']) == 1:
                    ttime = random.randint(5,10)
                elif int(previous_node['grade']) == 2:
                    ttime = random.randint(20,30)
                else:
                    ttime = random.randint(40,50)
            else:
                ttime = 14
            flow += ttime
    return flow

if __name__ == '__main__':
    with open('road_meta.json','r') as f:
        config = json.load(f)

    target = config['parking_locations']
    records_folder = 'dense_records'
    records = []

    for result_file in os.listdir(records_folder):
        result_path = os.path.join(records_folder,result_file)
        with open(result_path,'r') as f:
            data = json.load(f)
        logger.info(f'Total requirements: {len(data)}')
        record_time = datetime.datetime.fromtimestamp(int(result_file.split('.')[0].split('-')[-1])).strftime('%Y-%m-%d|%H')
        record_day,record_hour = record_time.split('|')
        for i in range(len(data)):
            s = data[i]
            ts = s['target']
            for k in ts:
                routes = ts[k]
                for r in routes:
                    distance = r['distance']
                    traffic_flow = analyze_flow(r['utic'])
                    records.append(
                        dict(start_lat=s['location'][0],start_long=s['location'][1],start_index=i,
                            end_lat=target[int(k)][0],end_long=target[int(k)][1],park_index=int(k),
                            day=record_day,record_hour=record_hour,
                            distance_m=distance,time_travel_s=traffic_flow)
                    )

    df = pd.DataFrame.from_dict(records)
    df.to_csv('records.csv')
    print("Complete!")

0
