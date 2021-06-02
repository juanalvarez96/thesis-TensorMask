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
from detectron2.data.datasets import register_coco_instances

coco_path_ann = "/home/juan.vallado/data/sequences_sampled/20190520_155011.json"
img_path = "/home/juan.vallado/data/sequences_sampled/20190520_155011"
# Prepare dataloader for transformations
register_coco_instances("seabirds_train", {}, coco_path_ann, img_path)
seabirds_metadata = MetadataCatalog.get("seabirds_train")
datset=DatasetCatalog.get("seabirds_train")
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
cfg.DATASETS.TRAIN = ("seabirds_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 6200    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.SOLVER.MAX_ITER = 6200   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
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
dataset_dicts = datset
imgs=[]
folder = "/home/juan.vallado/data/sequences_sampled/20190520_155011/"
ims=os.listdir(folder)
for img in random.sample(ims, 10):
    file = os.path.join(folder, img)
    im = np.array(Image.open(file).convert('RGB'))
    im = im[:,:,::-1]# to bgr
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1],
        metadata=seabirds_metadata,
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

    
