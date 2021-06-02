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

# Only one sequence will be considered. Just one video.
# Paths with multiple images
annotation_path="/home/juan.vallado/data/YoutubeVIS/train/Annotations/0a8c467cc3"
image_path="/home/juan.vallado/data/YoutubeVIS/train/JPEGImages/0a8c467cc3"
#annotation_path="/home/juan.vallado/data/YoutubeVIS/train/Annotations/0a7a2514aa"
#image_path="/home/juan.vallado/data/YoutubeVIS/train/JPEGImages/0a7a2514aa"


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
            ann['bbox']=extract_bboxes(np.where(np.asarray(im, order="F")==i, i, 0), im)
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
    

def extract_bboxes(arr, im):
  # Extract class
  props = regionprops(arr)
  assert len(props) == 1, "Error: Expected one mask, but got {}".format(len(props))
  assert props[0].area > 0, "Error: Area of mask is <0!"
  rmin, cmin, rmax, cmax=props[0].bbox
  return [float(cmin), float(rmin), float(cmax-cmin), float(rmax-rmin)]
  #return props[0].bbox
get_youtube_dicts(image_path)

for d in ["train"]:
    DatasetCatalog.register("ytvis_" + d, lambda d=d: get_youtube_dicts(image_path))
    #MetadataCatalog.get("ytvis_" + d).thing_classes = ["sth", "another", 'sd', "caca"]
ytvis_metadata = MetadataCatalog.get("ytvis_train")

# Visualize annotations!
dataset_dicts = get_youtube_dicts(image_path)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=ytvis_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite("prueba.jpg", out.get_image()[:, :, ::-1])

# Prepare dataloader for transformations
datset=DatasetCatalog.get("ytvis_train")
dataloader = detectron2.data.build_detection_train_loader(dataset=datset,
    mapper = DatasetMapper(cfg = get_cfg(), is_train=True, augmentations=[
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.8),
        T.RandomRotation(angle=[315.0, 45.0], expand=False, sample_style="range")

]), total_batch_size=128
)

# RUN
cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ytvis_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
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
    cv2.imwrite("{}x.jpg".format(i), imgs[i])

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

    
