import os, shutil
import subprocess
import argparse
import json
import datetime
import pprint as pp
from utils import strip_abc
from meta import ParkingMeta, Meta
import pprint as pp

def file_path(string):
    if os.path.isfile(string):
        return string
    elif os.path.isdir(string):
        return string
    else:
        raise NotImplemented(string)

def get_video_meta(filepath):
    cmnd = ['ffprobe', '-show_format', '-pretty', '-loglevel', 'quiet' , '-print_format', 'json', '-show_entries',
            'stream=r_frame_rate','-select_streams', 'v:0', filepath]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(filepath)
    out, err =  p.communicate()
    print("==========output==========")
    out = out.decode('utf-8').replace('\n','')
    data_format = json.loads(out)
    pp.pprint(data_format)

    if err:
        print("========= error ========")
        print(err.decode('utf-8'))
        return None

    filename = data_format['format']['filename'].split('/')[-1]
    id = strip_abc(filename)
    init_time = data_format['format']['tags']['com.apple.quicktime.creationdate'][:-5]
    duration = data_format['format']['duration']
    fps = int(data_format['streams'][0]['r_frame_rate'].split('/')[0])/1000
    location = data_format['format']['tags']['com.apple.quicktime.location.ISO6709']

    video_meta = Meta(id=id,
                        filename = filename,
                        capture_time=init_time,
                        duration=duration,
                        fps=fps,
                        location=location)
    return video_meta

def extract_images_from_video(filepath,image_folder,everyframe=7):
    filename = get_video_name(filepath)
    os.system(f"ffmpeg -i {filepath} -vf 'select=not(mod(n\,{everyframe}))' -vsync vfr {os.path.join(image_folder,filename+'_'+'%06d.png')}")
    print('[DONE!] at ',image_folder)
    return filename

def get_video_name(filepath):
    return filepath.split('/')[-1].split('.')[0]

def get_imgs_meta(init_time,video_meta,everyframe=10):
    imgs = os.listdir(video_meta.image_folder)
    imgs_meta_list = [Meta(id=strip_abc(imgs[i]), reference_video_id=video_meta.id,
                        file_name=os.path.join(video_meta.image_folder,imgs[i]),
                        capture_time=(init_time + datetime.timedelta(seconds=i*everyframe/video_meta.fps)).strftime("%Y-%m-%dT%H:%M:%S.%f"))
                        for i in range(len(imgs))]
    return imgs_meta_list

def check_image_folder_collision(image_folder,override):
    if os.path.isdir(os.path.abspath(image_folder)):
        if override:
            # shutil.rmtree(os.path.abspath(image_folder))
            # os.makedirs(os.path.abspath(image_folder))
            return False
        else:
            print(f"Use existed images in {image_folder} without override")
            return True
    else:
        print("Creating image folder at", os.path.abspath(image_folder))
        os.makedirs(os.path.abspath(image_folder))
    print("Folder has been setup")
    return False

def export_meta_per_video(video_path,image_folder,meta,everyframe=7,is_extracted=False):
    video_meta = get_video_meta(video_path)
    assert video_meta, "Please handle errors first"
    video_meta.add(image_folder=image_folder)
    if not is_extracted:
        extract_images_from_video(video_path,video_meta.image_folder,everyframe)

    init_time = datetime.datetime.strptime(video_meta.capture_time, "%Y-%m-%dT%H:%M:%S")
    imgs_meta_list = get_imgs_meta(init_time,video_meta,everyframe)

    meta.add_meta("videos",video_meta)
    for img_meta in imgs_meta_list:
        meta.add_meta("images",img_meta)
    return meta


if __name__ == '__main__':
    """
        video should be a numberic name
        ex: 1.mp4, 21312.avi, 68.MOV
    """
    parser = argparse.ArgumentParser(description='Exporting data')
    parser.add_argument('--video', type=file_path, required=True,
                        help="video/videos' folder path")
    parser.add_argument('--meta', type=str, default='meta.json',
                        help="generated meta.py meta file's path")
    parser.add_argument('-f', type=str, default='Images',
                        help="images_folder path")
    parser.add_argument('-o', action='store_true',
                        help="override image foler")
    args = parser.parse_args()

    meta = ParkingMeta(meta_path=args.meta)

    if os.path.isdir(args.video):
        for v in os.listdir(args.video):
            image_folder = os.path.join(args.f,str(strip_abc(v)))
            is_extracted = check_image_folder_collision(image_folder,args.o)
            # print(image_folder,is_extracted)
            meta = export_meta_per_video(os.path.join(args.video, v),image_folder,meta,everyframe=10,is_extracted=is_extracted)
    else:
        is_extracted = check_image_folder_collision(args.f,args.o)
        meta = export_meta_per_video(args.video,args.f,meta,is_extracted=is_extracted)
    # pp.pprint(meta.meta)
    meta.export_meta_as_json()
