"""
실행 ex)
> python jaywalk_detectv2.py --data_dir ./imgset --result_dir ./result

"""
import os, sys, glob
import argparse
from ultralytics import YOLO
import roboflow
import torch
import numpy as np
import json
from PIL import Image
import cv2

torch.set_printoptions(sci_mode=False)

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help=' : Please set the data directory')
parser.add_argument('--result_dir', required=True, help=' : Please set the result directory', default='result/')

args = parser.parse_args()

#model
model = YOLO('./runs/segment/train6/weights/best.pt')

#main
def main(argv, args):
    print(f'argv : ', argv)
    print(f'args : ', args)
    print(f'args.data_dir : ', args.data_dir)
    print(f'args.data_dir : ', args.result_dir)

    file_pathes = glob.glob(os.path.join(args.data_dir,'*.jpg'))
    print(file_pathes)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    for file_path in file_pathes:
        file_name, _ = os.path.splitext(os.path.basename(file_path))
        json_dir = args.result_dir+'/json'

        result_bboxes = []
        
        prediction=model.predict(file_path, conf=0.4)
        data = prediction[0].boxes.data
        label_list = data[data[:, 5] == 1]
        width = prediction[0].orig_img.shape[1]
        height = prediction[0].orig_img.shape[0]
                                         
                         
        seg_json = open(os.path.join(json_dir, file_name+'.json'), 'w')
        seg_json.write(prediction[0].tojson())
        seg_json.close()

        with open(os.path.join(json_dir, file_name+'.json'), 'r') as f:
            data = json.load(f)


        road_segments = []
        mask = np.zeros(prediction[0].orig_shape, dtype=np.uint8)
        for element in data:
            if element['name'] == 'road':
                x_y_pairs = list(zip(element['segments']['x'], element['segments']['y']))
                road_segments.append(x_y_pairs)
        road_segments = [np.array(segment, dtype=np.int32) for segment in road_segments]
        cv2.fillPoly(mask, road_segments, color=255)
        
        result_txt = open(os.path.join(args.result_dir, file_name+'.txt'), 'w')
        #pedestrian bbox
        for i, box_info in enumerate(label_list):
            
            x1, y1, x2, y2 = map(int, box_info[:4])  # 바운딩 박스의 좌표 추출
            threshold = 15
            norm_x1, norm_y1, norm_x2, norm_y2 = x1/width, y1/height, x2/width, y2/height
            center_x = norm_x1 + (norm_x2-norm_x1)/2
            center_y = norm_y1 + (norm_y2-norm_y1)/2
            bbox_width = norm_x2-norm_x1
            bbox_height = norm_y2-norm_y1
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=5)
            bottom_x = int((x2-x1) / 2 + x1)
            bottom_y = int(y2) + threshold
            if bottom_y > height:
                bottom_y = bottom_y - threshold
            
            print(i, bottom_y + threshold,bottom_x, mask[int(y1+(y2-y1)/2)][int(x1+(x2-x1)/2)])
            #cv2.circle(mask, (bottom_y + threshold, bottom_x), radius=50,color=0,thickness=2)
            if mask[bottom_y + threshold][bottom_x] > 0:
                result_txt.write(f"0 {center_x} {center_y} {bbox_width} {bbox_height}\n")
            else:
                result_txt.write(f"1 {center_x} {center_y} {bbox_width} {bbox_height}\n")
        #cv2.imwrite(file_name+'.jpg', mask)

        result_txt.close()
            
            




if __name__ == '__main__' :
    argv = sys.argv
    main(argv, args)


