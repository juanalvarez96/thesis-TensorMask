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
from detectron2.data import MetadataCatalog, DatasetCatalog
import pycocotools
from PIL import Image, ImageDraw
import numpy as np
import ipdb
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
import re

from detectron2.evaluation import COCOEvaluator


# Only one sequence will be considered. Just one video.
# Paths with multiple images
parent_annotation_path="/home/juan.vallado/data/YoutubeVIS/train/Annotations/"
parent_image_path="/home/juan.vallado/data/YoutubeVIS/train/JPEGImages/"
coco_path = "/home/juan.vallado/data/coco/train2017"
#annotation_path="/home/juan.vallado/data/YoutubeVIS/train/Annotations/0a7a2514aa"
#image_path="/home/juan.vallado/data/YoutubeVIS/train/JPEGImages/0a7a2514aa"

# Get crowds within ytvis given a threshold of minimum objects
def get_crowds(threshold):
    data = json.load(open("/home/juan.vallado/data/YoutubeVIS/train/train.json"))
    inf = {}

    for el in data['annotations']:
        inf.setdefault(int(el['video_id']), []).append(int(el['category_id']))

    # Filter to only those ids with crowds and one category
    new_inf = {}
    for (key, values) in inf.items():
        if (len(set(values))==1 and len(values)>threshold):
            new_inf[key]=values
    # Get video name folder
    videos = []
    for video in data['videos']:
        if video['id'] in new_inf.keys():
            videos.append(str(video['file_names'][0][:10]))
    return videos

# Select one candidate and delete from candidate list
candidates = get_crowds(4)
dataset = random.choice(candidates)
candidates.remove(dataset)
# Igual tenemos que cambiar todo lo que llama PIL
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

def transformations(name, cfg):
    datset=DatasetCatalog.get(name)
    dataloader = detectron2.data.build_detection_train_loader(dataset=datset,
        mapper = DatasetMapper(cfg = cfg, is_train=True, augmentations=[
            T.RandomBrightness(0,1, 10),
            T.RandomFlip(prob=0.5),
            T.RandomRotation(angle=[315.0, 45.0], expand=False, sample_style="range"),
            T.RandomContrast(0,1, 10),
            T.RandomExtent((100, 50), (100, 80)),
            T.RandomSaturation(0.1,10),
            T.RandomLighting(8),
            T.RandomCrop("relative_range", (0.1,1))

]), total_batch_size=128)
annotation_path = os.path.join(parent_annotation_path, dataset)
image_path = os.path.join(parent_image_path, dataset)

for d in ["train", "test"]:
        DatasetCatalog.register("ytvis_" + d, lambda d=d: get_youtube_dicts("{0}/{1}".format(image_path, d), "{0}/{1}".format(annotation_path, d)))
        MetadataCatalog.get("ytvis_" + d).thing_classes = ["la_cosa", "nada"]
ytvis_metadata = MetadataCatalog.get("ytvis_train")

def visualize(n):
    dataset_dicts = DatasetCatalog.get("ytvis_train")
    imgs = []
    for d in random.sample(dataset_dicts, n):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=ytvis_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        imgs.append(out.get_image()[:, :, ::-1])

    for i in range (0, len(imgs)):
        cv2.imwrite("{}x.jpg".format(i), imgs[i])
# Uncomment to see results data preprocessing
#visualize(8) 

# Implement trainer module to use coco validation during training
class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("/home/appuser/output/training_eval", exist_ok=True)
        output_folder = "/home/appuser/output/training_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

# START TRAINING

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ytvis_train",)
cfg.DATASETS.TEST = ("ytvis_test", )
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 2000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.INPUT.MASK_FORMAT="bitmask"

output_name= os.path.join( \
    "/home/appuser/output", \
    "{}".format(dataset)
)
cfg.OUTPUT_DIR = output_name
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
# Dump config
with open('{}/config.yaml'.format(output_name), 'a') as f:
        f.write(cfg.dump())   



# RETRAIN FOR TRANSFER LEARNING
dataset2 = random.choice(candidates)
# Register dataset
for d in ["train", "test"]:
        DatasetCatalog.register("ytvis2_" + d, lambda d=d: get_youtube_dicts("{0}/{1}".format(image_path, d), "{0}/{1}".format(annotation_path, d)))
        MetadataCatalog.get("ytvis2_" + d).thing_classes = ["la_cosa", "nada"]
ytvis_metadata = MetadataCatalog.get("ytvis2_train")

# Use same config as before:
cfg = get_cfg()
cfg.merge_from_file("/home/appuser/output/{}/config.yaml".format(dataset))

# Import weights
cfg.MODEL.WEIGHTS = os.path.join("/home/appuser/output/{}".format(dataset), "model_final.pth")  # path to the model we just trained

# Freeze later!
cfg.MODEL.BACKBONE.FREEZE_AT=4
cfg.DATASETS.TRAIN = ("ytvis2_train",)
cfg.DATASETS.TEST = ("ytvis2_test",)

# Generate output
output_name= os.path.join( \
    "/home/appuser/output", \
    "{}".format(dataset2)
)
cfg.OUTPUT_DIR = output_name
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Apply transformations for the second dataset
transformations("ytvis2_train", cfg)

# TRANSFER LEARNING
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
# Dump config
with open('{}/config.yaml'.format(output_name), 'a') as f:
        f.write(cfg.dump())   




