# multi-task-bdd100k_to_yolov5


Using (this)[https://www.bdd100k.com/] self driving data set to process labels of interest.

The module allows for laabel processing and filtering the data for classes of interest and task of interest. 

The exmaple included parses the data by:

1. Selecting a .json from the labels in the original download

2. Filters by specific classes that are of interest

3. Writes a csv with all the metadata available in the original data source

4. Parses the csv to change from COCO format of labels to YOLO format. 
