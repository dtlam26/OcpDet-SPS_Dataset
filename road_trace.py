import requests
import json
import numpy as np
import time
import concurrent.futures as cf
from loguru import logger

def mapquest_route(start_lat_long,target_lat_long):
    formData = {
        "locations": [
        	{
        		"latLng": {
        			"lat": float(start_lat_long[0]),
        			"lng": float(start_lat_long[1])
        		}
        	},
        	{
        		"latLng": {
        			"lat": float(target_lat_long[0]),
        			"lng": float(target_lat_long[1])
        		}
        	}
        ],
        "maxRoutes": 5,
        "options": {
        	"avoids": [],
            "conditionsAheadDistance": 321.87,
        	"doReverseGeocode": False,
        	"enhancedNarrative": True,
        	"narrativeType": "microformat",
        	"routeType": "fastest",
        	"shapeFormat": "cmp6",
            "unit": "k"
        },
        "timeOverage": 99
    }

    url = 'https://www.mapquest.com/route?key=Cmjtd%7Cluur2108n1%2C7w%3Do5-gz8a&timeType=1'
    x = requests.post(url, json=formData).json()
    return [{'distance': round(x['route']['distance'],6),
        'encodedPoints': x['route']['shape']['shapePoints']}]
#mapquest ASCII decode
def decompress(encodedPoints, precision=6):
    precision = 10**(-precision)
    l = len(encodedPoints)
    index=0
    lat=0
    lng = 0
    array = []

    def decode(char):
        return ord(char) - 63

    while (index < l):
        shift = 0
        result = 0
        b = decode(encodedPoints[index])

        while (True):
            result |= (b & 0x1f) << shift
            index += 1
            if b < 0x20:
                break
            shift += 5
            b = decode(encodedPoints[index])
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        shift = 0
        result = 0
        b = decode(encodedPoints[index])
        while (True):
            result |= ((b & 0x1f) << shift)
            index += 1
            if b < 0x20:
                break
            shift += 5
            b = decode(encodedPoints[index])
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng
        array.append([lat * precision,lng * precision])
    return array
#latLng to box cordinate
def utic_bbox_encode(lat,long):
    url = f'http://dapi.kakao.com/v2/local/geo/transcoord.json?x={long}&y={lat}&input_coord=WGS84&output_coord=WTM'
    headers = {
        'Host': 'dapi.kakao.com',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:101.0) Gecko/20100101 Firefox/101.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'KA': 'sdk/4.4.3 os/javascript lang/en-US device/Linux_x86_64 origin/http%3A%2F%2Fwww.utic.go.kr',
        'Authorization': 'KakaoAK 6e0344e0e6e27e6d6ee7158fe54f84b8',
        'Origin': 'http://www.utic.go.kr',
        'Connection': 'keep-alive',
        'Referer': 'http://www.utic.go.kr/',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache'
    }
    x=json.loads(requests.get(url,headers=headers).content.decode('utf-8'))
    return list(x['documents'][0].values())

def find_region_info(center_point):
    def decode_utic(respond_string):
        preior = "\'"
        results = []
        for x in respond_string[1:-1].split(','):
            x.replace("[^a-zA-Z0-9]", "")
            c = x.split(':')
            c[0] = c[0].replace(" ","")
            c[0] = f"{preior}{c[0]}{preior}"
            results.append((':').join(c))
        new_string = ','.join(results)
        new_string = new_string.replace("\'", "\"")
        return json.loads('{'+new_string+'}')

    width=850
    height=850
    #ratio prefix
    ratio=np.asarray([2.25368821e-06,2.25368821e-06])

    x = int(width/2)
    y = int(height/2)

    #abstract: the road location at center of a square map (850,850) with the lowest scale
    center_point = np.asarray(center_point)
    top_right = center_point+np.asarray([y,x])*ratio
    top_right = utic_bbox_encode(top_right[0],top_right[1])
    bottom_left = center_point-np.asarray([y,x])*ratio
    bottom_left = utic_bbox_encode(bottom_left[0],bottom_left[1])
    url = f"http://www.utic.go.kr/map/getproxy.do?url=http://61.108.209.20/getTrafficInfo&BBOX={bottom_left[0]}%2C{bottom_left[1]}%2C{top_right[0]}%2C{top_right[1]}&WIDTH={width}&HEIGHT={height}&SHAPE_LAYER=UTIS%3AP_LV1_D&X={x}&Y={y}"

    res = requests.get(url).content.decode('utf-8')
    if res:
        res = decode_utic(res)
    else:
        res = {}
    return res

def query_route_info(info):
    start_lat_long = info[0]
    target = info[1]
    output = {'location': start_lat_long,'target':{}}
    j = -1

    for target_lat_long in target:
        logger.info(f"{start_lat_long} => {target_lat_long}")
        j = j + 1
        try:
            routes = mapquest_route(start_lat_long,target_lat_long)
        except:
            logger.info("Can't find path")
            continue

        #Decode Routes
        for r in routes:
            r['components'] = decompress(r['encodedPoints'],6)
            r['utic'] = []

        #Utic Points
        for r in routes:
            for c in r['components']:
                res = find_region_info(c)
                r['utic'].append(res)

        output['target'][j] = routes
    return output

if __name__ == '__main__':
    with open('road_meta.json','r') as f:
        config = json.load(f)

    start = config['road_locations']
    target = config['parking_locations']

    i = 0
    j = 0

    with cf.ProcessPoolExecutor() as executor:
        collect_results = executor.map(query_route_info,[[start[i],target] for i in range(len(start))])


    with open(f'results-{int(time.time())}.json','w') as f:
        json.dump(list(collect_results), f,  indent=4)
    logger.info("COMPLETED!")
