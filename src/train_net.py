#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TensorMask Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import torch
import os, json, cv2, random
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops, find_contours
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

# Some imports for the custom data
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from tensormask import add_tensormask_config
from PIL import Image
import pycocotools
import detectron2

# Debug
import ipdb

annotation_path="/home/juan.vallado/data/YoutubeVIS/train/Annotations/0a8c467cc3"
image_path="/home/juan.vallado/data/YoutubeVIS/train/JPEGImages/0a8c467cc3"

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

def register_custom():
    for d in ["train", "test"]:
        DatasetCatalog.register("ytvis_" + d, lambda d=d: get_youtube_dicts("{0}/{1}".format(image_path, d), "{0}/{1}".format(annotation_path, d)))
        MetadataCatalog.get("ytvis_" + d).thing_classes = ["la_cosa"]


def transformations():
    datset=DatasetCatalog.get("ytvis_train")
    dataloader = detectron2.data.build_detection_train_loader(dataset=datset,
        mapper = DatasetMapper(cfg = get_cfg(), is_train=True, augmentations=[
            T.RandomBrightness(0.9, 1.1),
            T.RandomFlip(prob=0.8),
            T.RandomRotation(angle=[315.0, 45.0], expand=False, sample_style="range")

]), total_batch_size=128)

def visualize(n):
    from detectron2.utils.visualizer import Visualizer
    dataset_dicts = DatasetCatalog.get("ytvis_train")
    for d in random.sample(dataset_dicts, n):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite("{}.jpg".format(d['image_id']),out.get_image()[:, :, ::-1])

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tensormask_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    torch.cuda.empty_cache()
    # Preparation for custom dataset
    register_custom()
    transformations()
    #ipdb.set_trace()
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
