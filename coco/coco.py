'''
file : COCO.py

author : Aung Paing
cdate : Wednesday September 27th 2023
mdate : Wednesday September 27th 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
'''
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Optional, Union

import cv2
import numpy as np

COLOR_PALETTE = [
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 128, 128), # Gray
    (192, 192, 192), # Silver
    (128, 0, 0),     # Maroon
    (128, 128, 0),   # Olive
    (0, 128, 0),     # Green (Dark)
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (0, 64, 128),    # Blue (Dark)
    (64, 0, 128),    # Purple (Dark)
    (128, 64, 0),    # Brown
    (0, 128, 64),    # Green (Light)
    (0, 64, 0),      # Green (Dark)
    (64, 0, 0),      # Red (Dark)
    (255, 128, 128),  # Light Red
    (255, 128, 0),   # Orange
    (255, 255, 128),  # Light Yellow
    (128, 255, 128),  # Light Green
    (128, 255, 255),  # Light Cyan
    (128, 128, 255),  # Light Blue
    (255, 128, 255),  # Light Purple
    (192, 128, 64),  # Tan
    (255, 192, 128),  # Peach
    (255, 192, 192),  # Light Pink
    (128, 128, 64),  # Olive (Dark)
    (128, 64, 64),   # Brown (Dark)
    (128, 64, 128),  # Purple (Medium)
    (128, 0, 64),    # Maroon (Dark)
    (64, 128, 0),    # Olive (Light)
    (192, 64, 0),    # Orange (Dark)
    (192, 192, 0),   # Yellow (Dark)
    (192, 192, 128),  # Khaki
    (192, 192, 64),  # Olive (Medium)
    (128, 192, 64),  # Olive (Medium-Dark)
    (192, 128, 128),  # Rose
    (64, 192, 128),  # Green (Medium-Light)
    (64, 192, 192),  # Teal (Medium)
    (64, 64, 192),   # Blue (Medium)
    (192, 64, 192),  # Purple (Medium-Light)
    (192, 192, 192),  # Light Gray
    (0, 0, 64),      # Blue (Dark)
    (0, 64, 64),     # Teal (Dark)
    (0, 0, 128),     # Blue (Dark)
    (0, 128, 128),   # Teal (Medium)
    (0, 64, 192),    # Blue (Medium-Light)
    (64, 0, 64),     # Purple (Dark)
    (64, 64, 0),     # Olive (Dark)
    (64, 0, 0),      # Maroon (Dark)
    (192, 0, 0),     # Red (Medium-Dark)
    (192, 0, 192),   # Purple (Medium-Dark)
    (192, 128, 0),   # Brown (Medium)
    (128, 128, 192),  # Blue (Medium-Light)
    (128, 0, 192),   # Purple (Medium-Dark)
    (192, 0, 128),   # Magenta (Dark)
    (128, 0, 64),    # Maroon (Medium-Dark)
    (128, 64, 192),  # Purple (Medium-Light)
    (64, 128, 192),  # Blue (Medium-Light)
    (192, 128, 192),  # Pink (Light)
    (192, 0, 64),    # Red (Medium-Dark)
    (192, 64, 128),  # Pink (Medium)
    (64, 192, 0),    # Green (Medium)
    (128, 192, 128),  # Green (Medium-Light)
    (128, 192, 192),  # Cyan (Medium)
    (192, 192, 128),  # Green (Medium-Light)
    (192, 192, 0),   # Yellow (Medium)
    (0, 192, 64),    # Green (Medium-Light)
    (0, 192, 192),   # Teal (Medium)
    (64, 192, 192),  # Teal (Medium-Light)
    (128, 128, 64),  # Olive (Medium-Dark)
    (0, 128, 192),   # Blue (Medium-Light)
    (192, 128, 64),  # Brown (Medium)
    (192, 128, 0),   # Brown (Medium)
    (128, 128, 0),   # Olive (Medium)
    (64, 128, 64),   # Olive (Medium)
    (192, 64, 64),   # Red (Medium-Dark)
    (0, 128, 64),    # Green (Medium)
    (64, 192, 0),    # Green (Medium)
    (128, 64, 64),   # Red (Medium-Dark)
    (64, 64, 192),   # Blue (Medium-Light)
    (128, 0, 192),   # Purple (Medium-Dark)
    (64, 192, 128),  # Teal (Medium-Light)
    (128, 192, 192),  # Cyan (Medium-Light)
]


def assert_file(file_name: str):
    assert osp.exists(file_name), f"{file_name} not exists"


class BoundingBox:
    def __init__(self,
                x1: Union[int, float],
                y1: Union[int, float],
                w: Union[int, float],
                h: Union[int, float]):
        """COCO Format Bounding Box Class

        Args:
            x1 (Union[int, float]): Upper Left X coordinate
            y1 (Union[int, float]): Upper Left Y coordinate
            w (Union[int, float]): Width of Bounding Box
            h (Union[int, float]): Height of Bounding Box
        """
        self.__x1 = x1
        self.__y1 = y1
        self.__w  = w
        self.__h  = h
        self.__x2 = self.x1 + w
        self.__y2 = self.y1 + h
        self.__area = w * h

    def __repr__(self):
        return f"xyxy : [{self.x1:.2f},{self.y1:.2f},{self.x2:.2f},{self.y2:.2f}]"

    @property
    def x1(self):
        return self.__x1

    @property
    def x2(self):
        return self.__x2

    @property
    def y1(self) -> Union[int, float] :
        return self.__y1

    @property
    def y2(self) -> Union[int, float] :
        return self.__y2

    @property
    def w(self) -> Union[int, float] :
        return self.__w

    @property
    def h(self) -> Union[int, float] :
        return self.__h

    @property
    def area(self) -> Union[int, float] :
        return self.__area

    @property
    def xywh(self) -> list[Union[int, float]]:
        return [self.x1, self.y1, self.w, self.h]

    @property
    def xyxy(self) -> list[Union[int, float]]:
        return [self.x1, self.y1, self.x2, self.y2]

    def get_iou(self, other):
        _intersection_x1 = max(self.x1, other.x1)
        _intersection_y1 = max(self.y1, other.y1)
        _intersection_x2 = min(self.x2, other.x2)
        _intersection_y2 = min(self.y2, other.y2)

        _intersection = max( _intersection_x2 - _intersection_x1 , 0) * max(
            _intersection_y2 - _intersection_y1, 0
        )
        _union = (self.area + other.area) - _intersection
        iou = _intersection / _union
        return iou


class COCO:
    def __init__(self, annotation_file: str):
        self.file_name = annotation_file
        self.data      = self.read_json(self.file_name)
        self.cats      = self._loadIndex(self.data['categories'])
        self.imgs      = self._loadIndex(self.data['images'])
        self.annos     = self._loadIndex(self.data['annotations'])

        self._img_anno, self._cat_img = self._get_image_annotation_pair()

    def __repr__(self):
        coco_str = "COCO Dataset format annotation\n"
        coco_str += f"Number of Categories :\t {len(self.cats)}\n"
        coco_str += f"Number of Image : \t\t{len(self.imgs)}\n"
        coco_str += f"Number of Annotations :\t {len(self.annos)}"
        return coco_str

    @staticmethod
    def read_json(file_name: str):
        assert_file(file_name)
        return json.load(open(file_name, 'r'))

    @staticmethod
    def _loadIndex(annotation_list_object: list[dict]) -> dict[int, dict]:
        ret = {}
        for list_object in annotation_list_object:
            ret[list_object['id']] = list_object
        return ret

    def _get_image_annotation_pair(self):
        img_anno = defaultdict(list)
        cat_img = defaultdict(list)

        for annoId, annotation in self.annos.items():
            imgId = annotation['image_id']
            catId = annotation['category_id']

            img_anno[imgId].append(annoId)
            cat_img[catId].append(imgId)
        return img_anno, cat_img

    def getImgIds(self, catIds: Optional[list[int]] = None):
        imgIds = []
        if catIds is None:
            imgIds = [imgId for imgId, img in self.imgs.items()]
        else:
            imgIds = self._cat_img[catIds]
        return imgIds

    def getAnnIds(self, imgIds: Union[list[int], int]) -> list[int]:
        """Get all annotations ID for the given Image ID

        Args:
            imgIds (list[int]): imgID in the annotation
        Returns:
            annIds (list[int]): Annotation IDS for the imgIDs
        """
        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        annIds = []
        for imgId in imgIds:
            annIds += self._img_anno[imgId]
        return annIds

    def loadImgs(self, imgIds: Union[list[int], int]):
        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        imgs = []
        for imgId in imgIds:
            imgs.append(self.imgs[imgId])
        return imgs

    def loadAnns(self, annIds: Union[list[int], int]):
        annIds = annIds if isinstance(annIds, list) else [annIds]
        annos = []
        for annId in annIds:
            annos.append(self.annos[annId])
        return annos


class AssertCOCO:
    def __init__(self, coco: COCO):
        self.coco = coco

    def _assert_images(self, img_dir: str):
        """Assert the exist of image file name and the file is valid

        Args:
            img_dir (str): The base image dir name
        """
        assert_file(img_dir)
        for imgId, img in self.coco.imgs.items():
            img_base_name = img['file_name']
            img_full_name = osp.join(img_dir, img_base_name)
            assert_file(img_full_name)
            cv2.imread(img_full_name)

    def _assert_annotations_iou(self, iou_threshold=0.5):
        """Assert the IOU of the annotation in the images
        """
        for imgId, _ in self.coco.imgs.items():
            annIds = self.coco.getAnnIds(imgIds=imgId)
            img = self.coco.loadImgs(imgIds=[imgId])[0]
            img_base_name = img['file_name']

            for i, annId in enumerate(annIds):
                anno = self.coco.loadAnns(annIds=annId)[0]
                bbox = BoundingBox(*anno['bbox'])
                for j, otherAnnId in enumerate(annIds[i+1:]):
                    otherAnno = self.coco.loadAnns(annIds=otherAnnId)[0]
                    otherBbox = BoundingBox(*otherAnno['bbox'])
                    iou = bbox.get_iou(otherBbox)
                    assert iou < iou_threshold, f"{img_base_name} has \
                        \nIOU : {iou},\nBBox : {bbox} && {otherBbox}"

    def assert_anno_level_annotations(self):
        """Assert the correctness of annotation in COCO format
        For each annotation assert
        - If image ID is in the image ID list.
        - If category ID is in the category ID list.
        - Bounding box annotation is positive and not out of image shape
        """
        _imgIds = [imgId for imgId, _ in self.coco.imgs.items()]
        _catIds = [catId for catId, _ in self.coco.cats.items()]
        for _, anno in self.coco.annos.items():
            _imgId = anno['image_id']
            _catId = anno['category_id']

            # Assert bounding box correctness
            # Get current annotation bounding box coordinate
            _x1, _y1, _w, _h = anno['bbox']
            _x2, _y2 = _x1 + _w, _y1 + _h
            # Get image shape for this annotation
            _img = self.coco.imgs[_imgId]
            imgH, imgW = _img['height'], _img['width']

            assert 0 <= _x1 <= imgW, f"bbox coordinate out of range: {_x1} / {imgW}"
            assert 0 <= _x2 <= imgW, f"bbox coordinate out of range: {_x2} / {imgW}"
            assert 0 <= _y1 <= imgH, f"bbox coordinate out of range: {_y1} / {imgH}"
            assert 0 <= _y2 <= imgH, f"bbox coordinate out of range: {_y2} / {imgH}"

            assert _imgId in _imgIds, f"Image ID :{_imgId} is not correct"
            assert _catId in _catIds, f"Category ID :{_imgId} is not correct"

    def assert_img_level_annotations(self, img_dir: str):
        """Assert the correctness of annotation in image level
        For each image assert
        - Image file Exists
        - Image file readable ( not broken image file )
        - Annotations in the image has IOU less than 0.6

        Args:
            img_dir (str): The base image dir name
        """
        self._assert_images(img_dir)
        self._assert_annotations_iou(0.8)


class COCOVis:
    def __init__(self, 
                coco: COCO, 
                img_dir: str, 
                dst_dir: str, 
                COLOR_PALETTE: list[list[int]] = COLOR_PALETTE):
        self.__coco = coco
        self.img_dir = img_dir
        self.dst_dir = dst_dir
        self.COLOR_PALETTE = COLOR_PALETTE
        self.__font = cv2.FONT_HERSHEY_SIMPLEX
        self.__text_size = cv2.getTextSize('sample', self.__font, 0.5, 1)[0]

    def vis_bbox(self, img_arr: np.ndarray, bbox: BoundingBox, catId: int):
        """Visualizaion of the COCO format annotation

        Args:
            img_arr (np.ndarray): Numpy image array
            bbox (BoundingBox): Boundingbox with coordinate information
            catId (int): CategryID

        Returns:
            vis_img_arr: Visualization of the copy of original image
        """
        vis_img_arr = img_arr
        _color = self.COLOR_PALETTE[catId]
        _color_np = np.array(_color)

        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        # Draw Bounding Box Rectangle
        cv2.rectangle(vis_img_arr, [x1, y1], [x2, y2], _color, 2)

        # Write Class Name
        text = self.__coco.cats[catId]['name']
        txt_color = (0, 0, 0) if np.mean(_color_np) > 122 else (255, 255, 255)
        txt_bk_color = (_color_np * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            vis_img_arr,
            (x1, y1 + 1),
            (x1 + self.__text_size[0] + 1, y1 + int(1.4 * self.__text_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            vis_img_arr, text, (x1, y1 + self.__text_size[1]), self.__font, 0.5, txt_color, 1
        )
        return vis_img_arr

    def vis(self, imgId: int):
        """Visualization of the COCO format for given image ID

        Args:
            imgId (_type_): _description_

        Returns:
            vis_img_arr: Visualization of the image
        """
        annIds = self.__coco.getAnnIds(imgIds=imgId)
        img = self.__coco.loadImgs(imgIds=[imgId])[0]
        img_base_name = img['file_name']
        img_full_name = osp.join(self.img_dir, img_base_name)
        assert_file(img_full_name)
        img_arr = cv2.imread(img_full_name)
        dst_file = osp.join(self.dst_dir, img_base_name)

        for annId in annIds:
            anno = self.__coco.loadAnns(annIds=[annId])[0]
            bbox = BoundingBox(*anno['bbox'])
            catId = anno['category_id']
            img_arr = self.vis_bbox(img_arr, bbox, catId)

        return img_arr
