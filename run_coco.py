"""
file : run_coco.py

author : Aung Paing
cdate : Monday October 2nd 2023
mdate : Monday October 2nd 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
"""
import os
import os.path as osp
import random

import cv2

from coco.coco import COCO, AssertCOCO, COCOVis
from coco.coco_report import COCOReport

anno_file = "/home/aung/Documents/data/csp_drone/annotations/instances_default.json"
img_dir = "/home/aung/Documents/data/csp_drone/train_images"
dst_dir = "/home/aung/Documents/data/csp_drone/vis"
os.makedirs(dst_dir, exist_ok=True)

vis_txt_bg_color = True
vis_txt_above_bbox = True

# set to [] if we don't want any attribute for visualization
vis_txt_attribute = []

assert_iou = False
num_images = 50


def main():
    coco = COCO(anno_file)

    coco_assert = AssertCOCO(coco)
    coco_assert.assert_img_level_annotations(img_dir, assert_iou)
    coco_assert.assert_anno_level_annotations()

    coco_vis = COCOVis(
        coco, img_dir, dst_dir, vis_txt_bg_color, vis_txt_above_bbox, vis_txt_attribute
    )

    imgIds = coco.getImgIds()
    # imgIds = sorted(imgIds)
    random.shuffle(imgIds)

    frame_idx = 0

    for imgId in imgIds:
        img = coco.loadImgs(imgIds=imgId)[0]
        img_base_name = img["file_name"]
        img_full_name = osp.join(img_dir, img_base_name)
        dst_full_name = osp.join(dst_dir, img_base_name)
        if not osp.exists(img_full_name):
            continue
        # print(img_full_name)
        img_arr = coco_vis.vis(imgId)
        cv2.imwrite(dst_full_name, img_arr)

        frame_idx += 1
        if frame_idx >= num_images:
            break

    coco_report = COCOReport(anno_file)

    # Area Distribution
    area_ax = coco_report.area_ax
    area_ax.set_title("Distribution")
    area_plot = coco_report.area_fig
    area_plot.savefig(osp.join(dst_dir, "val_area.png"))

    # Class Distribution
    cls_ax = coco_report.cls_distribution_ax
    cls_ax.set_title("Class Count Occurance")
    cls_plot = coco_report.cls_distribution_fig
    cls_plot.savefig(osp.join(dst_dir, "val_dist_plot.png"))

    # Heatmap
    heatmap = coco_report.heatmap.heatmap
    cv2.imwrite(osp.join(dst_dir, "heatmap.png"), heatmap)


if __name__ == "__main__":
    main()
