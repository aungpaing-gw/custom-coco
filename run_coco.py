'''
file : run_coco.py

author : Aung Paing
cdate : Monday October 2nd 2023
mdate : Monday October 2nd 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
'''
import os
import os.path as osp
import random

import cv2

from coco.coco import COCO, AssertCOCO, COCOVis
from coco.coco_report import COCOReport

anno_file = './annotations/instances_default.json'
img_dir = 'img'
dst_dir = 'vis'
os.makedirs(dst_dir, exist_ok=True)


def main():
    coco = COCO(anno_file)
    print(coco)

    # coco_assert = AssertCOCO(coco)
    # coco_assert.assert_img_level_annotations(img_dir)
    # coco_assert.assert_anno_level_annotations()

    coco_vis = COCOVis(coco, img_dir, dst_dir)

    # Let's visualize for 10 images
    imgIds = coco.getImgIds()
    random.shuffle(imgIds)

    for imgId in imgIds:
        img = coco.loadImgs(imgIds=imgId)[0]
        img_base_name = img['file_name']
        img_full_name = osp.join(img_dir, img_base_name)
        if not os.path.exists(img_full_name): continue
        print(img_full_name)
        img_arr = coco_vis.vis(imgId)
        cv2.imwrite(img_full_name, img_arr)

    coco_report = COCOReport(anno_file)

    # Area Distribution
    area_ax = coco_report.area_ax
    area_ax.set_title(f"test: Area Distribution")
    area_plot = coco_report.area_fig
    area_plot.savefig('test_area.png')

    # Class Distribution
    cls_ax = coco_report.cls_distribution_ax
    cls_ax.set_title(f"test: Class Count Occurance")
    cls_plot = coco_report.cls_distribution_fig
    cls_plot.savefig("test_dist_plot.png")

    # Heatmap
    heatmap = coco_report.heatmap.heatmap
    cv2.imwrite('heatmap.png', heatmap)


if __name__ == '__main__':
    main()
