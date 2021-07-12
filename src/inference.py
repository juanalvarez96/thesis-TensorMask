# Some basic setup:
import torch, torchvision
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops, find_contours

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
import pycocotools
from PIL import Image, ImageDraw
import numpy as np
import ipdb
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
import re
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# Only one sequence will be considered. Just one video.
# Paths with multiple images
parent_annotation_path="/home/juan.vallado/data/YoutubeVIS/train/Annotations/"
parent_image_path="/home/juan.vallado/data/YoutubeVIS/train/JPEGImages/"

output_dir = "/home/appuser/output"

# Get datasets
dataset1 = os.listdir('/home/appuser/output')[0]
dataset2 = os.listdir('/home/appuser/output')[1]
datasets = [dataset1, dataset2]

def get_youtube_dicts(img_dir, mask_dir):
    id = 0
    data = []
    for image in os.listdir(img_dir):
        #if id == 0:
        #    record['annotations']=[]
        mask_path = os.path.join(mask_dir, "{}.png".format(image[:-4]))
        im=Image.open(mask_path)
        record = {}
        record['file_name']=os.path.join(img_dir, image)
        record['image_id']=id
        record['height']=im.size[1]
        record['width']=im.size[0]
        ann = {}
        objects = list(np.unique(np.asarray(im)))
        stuff = []
        for i in objects[1:]:
            ann['bbox']=extract_bboxes(np.where(np.asarray(im, order="F")==i, i, 0), im)
            ann['bbox_mode']=BoxMode.XYWH_ABS
        #ipdb.set_trace()
            ann['segmentation']=pycocotools.mask.encode( \
            np.asfortranarray( \
            np.where(np.asarray(im, order="F")==i, i, 0) \
            .astype(np.uint8)
                )   
            ) 
            
            ann['category_id']=0
            stuff.append(ann)
            ann={}
            
        record['annotations']=stuff
        data.append(record)
        segm = []

        id = id+1
    return data
    

def extract_bboxes(arr, im):
  # Extract class
  props = regionprops(arr)
  assert len(props) == 1, "Error: Expected one mask, but got {}".format(len(props))
  assert props[0].area > 0, "Error: Area of mask is <0!"
  rmin, cmin, rmax, cmax=props[0].bbox
  return [float(cmin), float(rmin), float(cmax-cmin), float(rmax-rmin)]
  #return props[0].bbox

def register(annotation_path, image_path, name):
    for d in ["train", "test"]:
        DatasetCatalog.register("ytvis_{}_".format(name) + d, lambda d=d: get_youtube_dicts("{0}/{1}".format(image_path, d), "{0}/{1}".format(annotation_path, d)))
        MetadataCatalog.get("ytvis_{}_".format(name) + d).thing_classes = ["la_cosa"]


# Register datasets
def register_datasets():
    cfg = get_cfg()
    for dataset in datasets:
        cfg.merge_from_file("/home/appuser/output/{}/config.yaml".format(dataset))
        if cfg.MODEL.BACKBONE.FREEZE_AT == 2: # BL model
            annotation_path = os.path.join(parent_annotation_path, dataset)
            image_path = os.path.join(parent_image_path, dataset)
            register(annotation_path, image_path, "bl")
        else:
            annotation_path = os.path.join(parent_annotation_path, dataset)
            image_path = os.path.join(parent_image_path, dataset)
            register(annotation_path, image_path, "tl")
            # Declare model and prepare weights for inference
            cfg = get_cfg()
            cfg.merge_from_file("/home/appuser/output/{}/config.yaml".format(dataset))
            cfg.MODEL.WEIGHTS = os.path.join("/home/appuser/output/{}".format(dataset), "model_final.pth")  # path to the model we just trained
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    #ipdb.set_trace()
    return cfg





# Prueba

cfg = register_datasets()
from detectron2.utils.visualizer import ColorMode
predictor = DefaultPredictor(cfg)

# Results for first dataset
evaluator = COCOEvaluator("ytvis_bl_test", cfg, False, output_dir)
val_loader = build_detection_test_loader(cfg, "ytvis_bl_test")
inference_on_dataset(predictor.model, val_loader, evaluator)

# Results for second dataset
evaluator = COCOEvaluator("ytvis_tl_test", cfg, False, output_dir)
val_loader = build_detection_test_loader(cfg, "ytvis_tl_test")
inference_on_dataset(predictor.model, val_loader, evaluator)


dataset_dicts = DatasetCatalog.get("ytvis_tl_test")
ytvis_metadata = MetadataCatalog.get("ytvis_tl_train")

imgs = []
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=ytvis_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    imgs.append(out.get_image()[:, :, ::-1])

for i in range (0, len(imgs)):
    cv2.imwrite("{}y.jpg".format(i), imgs[i])

