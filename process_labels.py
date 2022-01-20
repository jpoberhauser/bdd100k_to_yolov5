import torch
import glob
import json
import pandas as pd

def box_corner_to_center(boxes):
    """ Convert from (x1, y1, x2, y2) to (center_x, center_y, width, height).
    bbox1 = [60.0, 45.0, 378.0, 516.0]
    new_bbox = box_corner_to_center(bbox1)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    w = x2-x1
    h = y2-y1
    boxes_res =  torch.stack((cx, cy, w, h, boxes[:, 4], boxes[:, 5]), axis =-1)# on the last available axis
    return boxes_res


def convert2relative(boxes):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    width, height = boxes[:, 4], boxes[:, 5]
    boxes_res =  torch.stack(( x/width, y/height, w/width, h/height), axis =-1)# on the last available axis
   
    return boxes_res

def write_yolo_label(img_path_inq, output_directory, all_images_wp):
    # Process labels and write as yolo format
    if img_path_inq in all_images_wp:# if image is in images_with_people
        filt_df = val_set[val_set['img_path'] == img_path_inq]
        coords_x1y1x2y2 = filt_df[['x1','y1','x2','y2','img_width', 'img_height']].values
        coords_x1y1x2y2_tens = torch.tensor(coords_x1y1x2y2)
        boxes_cxcywh = box_corner_to_center(coords_x1y1x2y2_tens)
        yolo_coords = convert2relative(boxes_cxcywh)

        label_name = f"{output_directory}/{img_path_inq.replace('.jpg', '.txt')}"
        lines = []
        with open(label_name, 'w') as f:
            for i in yolo_coords.numpy():
                line = f"0 {i[0]} {i[1]} {i[2]} {i[3]} \n"
                f.write(line)
    else:# if its not: 
        label_name = f"{output_directory}/{img_path_inq.replace('.jpg', '.txt')}"
        open(label_name, mode='a').close()

        
############
## Set run parameters
#############        
JSON_LABELS = "self_driving_data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
CLASSES_OF_INTEREST =  ['person', 'rider']
ALL_IMG_PATH = "/home/jpoberhauser/Desktop/actuate_misc/self_driving_data/bdd100k/images/100k/val"
DATA_TYPE = "val"
LABEL_OUTPUT_DIR = "/home/jpoberhauser/Desktop/actuate_misc/self_driving_data/labels_val"
############
## Read in the json that was downloaded
#############
# Opening JSON file
f = open(JSON_LABELS)
# returns JSON object as
# a dictionary
data = json.load(f)

############
## Process
#############

all_rows = []
for idx in range(len(data)):
    for i in data[idx]['labels']:
        if i['category'] in CLASSES_OF_INTEREST:
            #print(idx, i, data[idx]['name'], data[idx]['attributes'])
            img_path = data[idx]['name']
            time_of_day = data[idx]['attributes']['timeofday']
            weather = data[idx]['attributes']['weather']
            occluded = i['attributes']['occluded']
            bbox = i['box2d']
            width_box = bbox['x2']- bbox['x1']
            length_box = bbox['y2']- bbox['y1']
            area_box = width_box*length_box
            
            pil_im = Image.open(f"{ALL_IMG_PATH}/{img_path}")
            new_name = f"-- {time_of_day}, {weather}, occluded-{occluded}, {bbox}, {img_path}"
            row = [img_path, time_of_day, weather, occluded, bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'], width_box, length_box, area_box, pil_im.size[0], pil_im.size[1], i['category']]
            all_rows.append(row)
            
### Make DF with metadata
all_df = pd.DataFrame(all_rows, columns = ["img_path", "time_of_day", "weather", "occluded",'x1', 'y1', 'x2', 'y2', 'width_box', 'length_box', 'area_box', 'img_width', 'img_height', 'category'])
all_df['data_type'] = DATA_TYPE
print(all_df.shape)
all_df.to_csv("coco_formatted_labels.csv")




############
## Do we need to post-process labels to yolo format?
#############
val_set = pd.read_csv("coco_formatted_labels.csv)
all_images = glob.glob("{ALL_IMG_PATH}/*.jpg")
print(len(all_images))
                      
for index_ in range(len(all_imgs)):
    write_yolo_label(all_imgs[index_], LABEL_OUTPUT_DIR, all_images_wp)
    
