"""PyOCR"""

import pathlib
import numpy as np
import torch
import logging
from huggingface_hub import hf_hub_download
from pyocr.inference.predict_system import PredictSystem


class PyOCR:
    """PyOCR class"""

    def __init__(
        self,
        use_angle_cls=False,
        devices="auto",
        model_local_dir=None,
        needWarmUp=False,
        warmup_size=(640, 640),
        drop_score=0.5,
        **kwargs,
    ):
        self.config_default_dict = {
            "det_model_path": "PaddleOCR2Pytorch/ch_ptocr_v4_det_infer.pth",
            "rec_model_path": "PaddleOCR2Pytorch/ch_ptocr_v4_rec_infer.pth",
            "cls_model_path": "PaddleOCR2Pytorch/ch_ptocr_mobile_v2.0_cls_infer.pth",
            "det_model_config_path": "PaddleOCR2Pytorch/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml",
            "rec_model_config_path": "PaddleOCR2Pytorch/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml",
            "character_dict_path": "ppocr_keys_v1.txt",
        }
        self._modelFileKeys = [
            "det_model_path",
            "rec_model_path",
            "cls_model_path",
            "det_model_config_path",
            "rec_model_config_path",
            "character_dict_path",
        ]
        self._modelFilePaths = {
            key: kwargs.get(key, None) for key in self._modelFileKeys
        }
        if devices == "auto":
            self._use_gpu = True if torch.cuda.is_available() else False
        else:
            self._use_gpu = True if devices == "cuda" else False
        logging.info(f"Using device: {devices}")
        self._model_local_dir = model_local_dir
        if self._model_local_dir:
            self._load_local_file(self._modelFilePaths)
        else:
            self._download_file(self._modelFilePaths)
        self.ocr = PredictSystem(
            use_angle_cls=use_angle_cls,
            det_yaml_path=self._modelFilePaths["det_model_config_path"],
            det_model_path=self._modelFilePaths["det_model_path"],
            rec_yaml_path=self._modelFilePaths["rec_model_config_path"],
            rec_model_path=self._modelFilePaths["rec_model_path"],
            cls_model_path=self._modelFilePaths["cls_model_path"],
            rec_char_dict_path=self._modelFilePaths["character_dict_path"],
            drop_score=drop_score,
            use_gpu=self._use_gpu,
        )
        self.needWarmUp = needWarmUp
        self._warm_up(warmup_size) if self.needWarmUp else None

    def _load_local_file(self, fileDict):
        for key, val in fileDict.items():
            if not val:
                logging.warning(
                    f"Unspecified {key[:-5]}, using default value {self.config_default_dict[key]}"
                )
                fileDict[key] = pathlib.Path(
                    self._model_local_dir, self.config_default_dict[key]
                )
                if not fileDict[key].exists():
                    raise FileNotFoundError(f"File {fileDict[key]} not found.")
        logging.info(fileDict)

    def _download_file(self, fileDict):
        for key, val in fileDict.items():
            if not val:
                logging.warning(
                    f"Unspecified {key[:-5]}, using default value {self.config_default_dict[key]}"
                )
                fileDict[key] = hf_hub_download(
                    repo_id="pk5ls20/PaddleModel",
                    filename=self.config_default_dict[key],
                )
        logging.info(fileDict)

    def _warm_up(self, warmup_size):
        logging.info("Warm up started")
        assert (
            isinstance(warmup_size, (list, tuple)) and len(warmup_size) == 2
        ), "warmup_size must be tuple or list with 2 elems."
        img = np.random.uniform(0, 255, [warmup_size[0], warmup_size[1], 3]).astype(
            np.uint8
        )
        for i in range(10):
            _ = self.ocr(img)
        logging.info("Warm up finished")
