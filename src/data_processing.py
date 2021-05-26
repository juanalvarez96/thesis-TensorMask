# Some basic setup:
import torch, torchvision
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pycocotools
from PIL import Image
import numpy as np
import ipdb
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

# Only one sequence will be considered. Just one video.
# Paths with multiple images
annotation_path="/home/juan.vallado/data/YoutubeVIS/train/Annotations/0a8c467cc3"
image_path="/home/juan.vallado/data/YoutubeVIS/train/JPEGImages/0a8c467cc3"


# Igual tenemos que cambiar todo lo que llama PIL
def get_youtube_dicts(img_dir):
    id = 0
    data = []
    for image in os.listdir(img_dir):
        #if id == 0:
        #    record['annotations']=[]
        mask_path = os.path.join(annotation_path, "{}.png".format(image[:-4]))
        im=Image.open(mask_path)
        record = {}
        record['file_name']=os.path.join(image_path, image)
        record['image_id']=id
        record['height']=im.size[1]
        record['width']=im.size[0]
        ann = {}
        objects = list(np.unique(np.asarray(im)))
        stuff = []
        for i in objects[1:]:
            ann['bbox']=extract_bboxes(np.where(np.asarray(im, order="F")==i, i, 0))[0]
            ann['bbox_mode']=BoxMode.XYWH_ABS
        #ipdb.set_trace()
            ann['segmentation']=pycocotools.mask.encode( \
            np.asfortranarray( \
            np.where(np.asarray(im, order="F")==i, i, 0) \
            .astype(np.uint8)
                )   
            ) 
            
            ann['category_id']=i
            stuff.append(ann)
            ann={}
            
        record['annotations']=stuff
        data.append(record)
        segm = []

        id = id+1
    return data
    

def extract_bboxes(arr):
  bboxes = []
  for mask_n in np.unique(arr)[1:]:
    # Extract class
    mask=np.where(arr == mask_n, 1, 0)
    rows=np.any(mask, axis = 1)
    cols=np.any(mask, axis = 0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox=[float((cmin+cmax)/2), float((rmin+rmax)/2), float(cmax-cmin), float(rmax-rmin)]
    bboxes.append(bbox)
  return bboxes
get_youtube_dicts(image_path)

for d in ["train"]:
    DatasetCatalog.register("ytvis_" + d, lambda d=d: get_youtube_dicts(image_path))
    #MetadataCatalog.get("ytvis_" + d).thing_classes = ["sth", "another", "yahora", "que"]
ytvis_metadata = MetadataCatalog.get("ytvis_train")


# RUN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ytvis_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
#cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
cfg.INPUT.MASK_FORMAT="bitmask"

# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_youtube_dicts(image_path)
imgs=[]
ipdb.set_trace()
for d in random.sample(dataset_dicts, 4):
    im = np.array(Image.open(d['file_name']).convert('RGB'))
    im = im[:,:,::-1]# to bgr
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1],
            metadata=ytvis_metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    imgs.append(out.get_image()[:, :, ::-1])

for i in range (0, len(imgs)):
    cv2.imwrite("{}.jpg".format(i), imgs[i])

ipdb.set_trace()

'''
ino='/home/juan.vallado/data/YoutubeVIS/train/Annotations/0a8c467cc3/'
for image in os.listdir(ino):
    path = os.path.join(ino, image)
    print("With PIL:{}\nWith cv(Anydepth):{}\nWith cv GDAL:{}" \
    .format(
        len(np.unique(np.array(Image.open(path)))),
        len(np.unique(cv2.imread(path, cv2.IMREAD_ANYDEPTH))),
        len(np.unique(cv2.imread(path, cv2.IMREAD_LOAD_GDAL)))
    ))
'''

    
