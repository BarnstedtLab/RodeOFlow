"""
Author: @Jakob Faust
Date: 10.10.2026

Includes Lighning pose and Deeplabcut wrappers for predicting keypoints on face and body.
Includes automatic download of models.
"""

from abc import ABC, abstractmethod
import os
import numpy as np
import yaml
from contextlib import nullcontext
import torch
# Updated autocast import to the modern, device-agnostic version
from torch.amp import autocast 

import cv2
import pandas as pd
from ruamel.yaml import YAML
import urllib.request
from pathlib import Path
import zipfile
import shutil

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def download_models(models_dir):
    models_dir = Path(models_dir)
    expected_cfg = models_dir / "face" / "dlc" / "project_config.yaml"

    if not expected_cfg.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        download_file = models_dir.parent / "rodeoflow_models_temp.zip"

        print("Downloading models...")
        nextcloud_url = "https://umg.files.academiccloud.de/s/F2BAgZM4SBDJBtn/download"
        try:
            urllib.request.urlretrieve(nextcloud_url, download_file)
        except Exception as e:
            print(f"Download failed: {e}")
            return None

        print(f"Unzipping models to {models_dir.parent}...")
        with zipfile.ZipFile(download_file) as mf_models:
            mf_models.extractall(models_dir.parent)

        if download_file.exists():
            download_file.unlink()

        # Safely move contents from the extracted folder into models_dir without deleting models_dir itself
        nested_dir = models_dir.parent / "rodeoflow_models"
        if nested_dir.exists() and nested_dir != models_dir:
            for item in nested_dir.iterdir():
                target_item = models_dir / item.name
                if target_item.exists():
                    if target_item.is_dir():
                        shutil.rmtree(target_item)
                    else:
                        target_item.unlink()
                shutil.move(str(item), str(models_dir))
            nested_dir.rmdir()

        print("...done")

    return dict(
        cfg_body_dlc = models_dir / "body/dlc/project_config.yaml",
        model_body_dlc = models_dir / "body/dlc/dlc-models-pytorch/iteration-1/MouseBodySep9-trainset95shuffle3/train/snapshot-best-260.pt",
        cfg_body_lp = models_dir / "body/lp/config.yaml",
        model_body_lp = models_dir / "body/lp/epoch=79-step=20000-best.ckpt",
        cfg_face_dlc = models_dir / "face/dlc/project_config.yaml",
        model_face_dlc = models_dir / "face/dlc/dlc-models-pytorch/iteration-0/MouseFaceAug21-trainset95shuffle2/train/snapshot-best-170.pt",
    )


def detect_engine(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if (os.path.basename(cfg_path).startswith('pytorch_') 
        or cfg.get('engine') == 'pytorch'):
        return 'pytorch'
    return None


class KeypointEngine(ABC):
    @abstractmethod
    def load_model_cfg(self, model_cfg_path):
        ...
    
    @abstractmethod
    def detect_keypoints(self,
        video_file: str,
        out_dir: str,
        make_labeled_video: bool = False,
        overwrite: bool = False):
        ...


class LPDetector(KeypointEngine):
    def __init__(self, model_cfg_path, model_path):
        self.cfg = self.load_model_cfg(model_cfg_path)
        self.model_path = model_path
        self.device = get_device()
        print(f"using device: {self.device}")
    
    def load_model_cfg(self, model_cfg_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(model_cfg_path)
        cfg.model.backbone = cfg.model.get("backbone", "resnet50")
        cfg.model.pretrained = False
        return cfg
    
    def lp_csv_to_dlc_h5(self, csv_path, out_h5=None, scorer_name="heatmap_tracker"):
        raw = pd.read_csv(csv_path, dtype=str, low_memory=False)
        bodyparts_row = raw.iloc[0]
        coords_row    = raw.iloc[1]

        pred_cols, keep_cols = [], []
        for j, col in enumerate(raw.columns):
            if col == "scorer":
                continue
            bp  = str(bodyparts_row.iloc[j]).strip()
            crd = str(coords_row.iloc[j]).strip().lower() 
            pred_cols.append((scorer_name, bp, crd))
            keep_cols.append(col)

        df = raw.iloc[2:].copy()
        df["scorer"] = pd.to_numeric(df["scorer"], errors="coerce")
        df = df.set_index("scorer").rename_axis("frame")
        df = df[keep_cols].apply(pd.to_numeric, errors="coerce")
        df.columns = pd.MultiIndex.from_tuples(pred_cols, names=["scorer","bodyparts","coords"])

        if out_h5 is None:
            out_h5 = Path(csv_path).with_suffix(".h5")
        df.to_hdf(out_h5, key="df_with_missing", mode="w")
        return str(out_h5)

    def detect_keypoints(
        self,
        video_file: str,
        out_dir: str,
        make_labeled_video: bool = False,
        overwrite: bool = False,
        prediction_suffix: str = "",):

        import tempfile
        import warnings
        from lightning_pose.utils.predictions import (export_predictions_and_labeled_video,generate_labeled_video,)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        base = Path(video_file).stem + prediction_suffix
        preds_csv = out_path / f"{base}.csv"
        preds_h5 = out_path / f"{base}.h5"
        labeled_mp4 = out_path.parent / f"{base}_labeled.mp4"

        # 1. Generate predictions without rendering a labeled video.
        # Reuse an existing CSV or H5 unless overwrite=True.
        run_prediction = overwrite or not (preds_csv.is_file() or preds_h5.is_file())
        if run_prediction:
            autocast_enabled = (autocast(device_type=self.device) if self.device in ["cuda", "cpu"] else nullcontext())
            with torch.no_grad():
                with autocast_enabled:
                    export_predictions_and_labeled_video(
                        video_file=str(video_file),
                        cfg=self.cfg,
                        prediction_csv_file=str(preds_csv),
                        ckpt_file=self.model_path,
                        labeled_mp4_file=None,)
            if not preds_csv.is_file():
                raise FileNotFoundError(f"LP did not create the prediction CSV: {preds_csv}")
        # 2. Save analysis-ready H5 before any optional video rendering.
        if run_prediction or not preds_h5.is_file():
            self.lp_csv_to_dlc_h5(str(preds_csv),out_h5=preds_h5,)
        print(f"Tracking H5 ready: {preds_h5}")
        # 3. Render the optional labeled video from saved predictions.
        video_result = None
        if make_labeled_video:
            needs_video = (overwrite or not labeled_mp4.is_file() or labeled_mp4.stat().st_size == 0)
            if needs_video:
                try:
                    predictions = pd.read_hdf(preds_h5,key="df_with_missing",)
                    # Write to a temporary location so a failed render
                    # does not leave a partial final video.
                    with tempfile.TemporaryDirectory(prefix="rodeoflow_label_",dir=str(labeled_mp4.parent),) as temp_dir:
                        temporary_video = Path(temp_dir) / "labeled.mp4"
                        generate_labeled_video(
                            video_file=str(video_file),
                            preds_df=predictions,
                            output_mp4_file=str(temporary_video),
                            confidence_thresh_for_vid=0.1,
                            colormap="cool",)
                        if (not temporary_video.is_file() or temporary_video.stat().st_size == 0):
                            raise RuntimeError("LP returned without creating a nonempty video.")
                        temporary_video.replace(labeled_mp4)
                    video_result = str(labeled_mp4)
                    print(f"Labeled video saved: {labeled_mp4}")
                except Exception as exc:
                    warnings.warn(
                        f"Tracking H5 is saved at {preds_h5}, but labeled "
                        f"video creation failed: {type(exc).__name__}: {exc}. "
                        "Run again with overwrite=False to retry rendering "
                        "without repeating prediction.",
                        RuntimeWarning,
                        stacklevel=2,)
            else:
                video_result = str(labeled_mp4)
                print(f"Using existing labeled video: {labeled_mp4}")
        return(str(preds_csv) if preds_csv.is_file() else None, video_result,)

class DLCDetector(KeypointEngine):
    def __init__(self, model_cfg_path, model_path, shuffle=3):
        self.model_path = model_path
        self.model_cfg = model_cfg_path
        self.shuffle = shuffle
        self.pose_cfg = Path(os.path.dirname(self.model_path)).parent / "test" /  "pose_cfg.yaml"
        self.device = get_device()
        print(f"using device: {self.device}")

    def create_labeled_video(
        self,
        video_path: str,
        h5_path: str,
        pose_cfg_path: str,
        out_path: str | None = None,
        pcutoff: float = 0.1,
        radius: int = 3,
        thickness: int = 2,
        draw_names: bool = False
    ):
        """
        Minimal labeled video writer for DLC single-animal outputs.
        - video_path: path to the original video
        - h5_path: DLC predictions (.h5) with MultiIndex columns (scorer, bodypart, coord)
        - pose_cfg_path: test/pose_cfg.yaml containing metadata or joint names
        - out_path: output mp4 path (defaults to <video_basename>_labeled.mp4 next to input)
        - pcutoff: confidence threshold for drawing
        """
        # load predictions first
        df = pd.read_hdf(h5_path)
        if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
            raise ValueError("Unexpected H5 columns. Expect MultiIndex (scorer, bodypart, coord).")

        # load bodyparts from pose config with multi-key fallbacks
        bodyparts = []
        if os.path.exists(pose_cfg_path):
            try:
                yaml_loader = YAML(typ="safe")
                with open(pose_cfg_path, "r") as f:
                    pose_cfg = yaml_loader.load(f) or {}
            except Exception:
                pose_cfg = {}
                
            bodyparts = (
                pose_cfg.get("metadata", {}).get("bodyparts")
                or pose_cfg.get("bodyparts")
                or pose_cfg.get("all_joints_names")
                or []
            )

        # Fallback: extract unique bodypart names directly from HDF5 MultiIndex columns
        if not bodyparts:
            bodyparts = df.columns.get_level_values(1).unique().tolist()

        if not bodyparts:
            raise ValueError("Could not extract bodyparts list from pose_cfg.yaml or HDF5 output file.")

        # map: bodypart -> (x_col, y_col, l_col)
        # pick the first scorer layer present
        scorers = sorted(set([c[0] for c in df.columns]))
        scorer = scorers[0]

        def col(bp, coord):
            candidates = [(scorer, bp, coord)]
            if coord == "likelihood":
                candidates.append((scorer, bp, "likelihoods"))
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        triplets = []
        for bp in bodyparts:
            xC = col(bp, "x")
            yC = col(bp, "y")
            lC = col(bp, "likelihood")
            if xC and yC and lC:
                triplets.append((bp, xC, yC, lC))
            else:
                continue

        if not triplets:
            raise ValueError("No valid (x,y,likelihood) columns found for listed bodyparts.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        nframes_pred = len(df)
        nframes = min(nframes_vid, nframes_pred)

        if out_path is None:
            base, _ = os.path.splitext(video_path)
            out_path = base + "_labeled.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not open writer: {out_path}")

        def bp_color(i, total):
            hue = int(179 * i / max(1, total))
            col_bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0, :]
            return int(col_bgr[0]), int(col_bgr[1]), int(col_bgr[2])

        colors = {bp: bp_color(i, len(triplets)) for i, (bp, *_cols) in enumerate(triplets)}

        for idx in range(nframes):
            ok, frame = cap.read()
            if not ok:
                break
            row = df.iloc[idx]
            for bp, xC, yC, lC in triplets:
                x = row[xC]
                y = row[yC]
                l = row[lC]
                if pd.isna(x) or pd.isna(y) or pd.isna(l) or float(l) < pcutoff:
                    continue
                cx, cy = int(round(float(x))), int(round(float(y)))
                cv2.circle(frame, (cx, cy), radius, colors[bp], thickness, lineType=cv2.LINE_AA)
                if draw_names:
                    cv2.putText(frame, bp, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[bp], 1, cv2.LINE_AA)
            writer.write(frame)

        cap.release()
        writer.release()
        return out_path

    def load_model_cfg(self, model_cfg_path):
        return os.path.join(model_cfg_path, "config.yaml")
    
    def detect_keypoints(self,
        video_file, out_dir,
        make_labeled_video = False,
        overwrite = False):

        from deeplabcut.compat import analyze_videos
        from deeplabcut.core.engine import Engine

        os.makedirs(out_dir, exist_ok=True)
        autocast_enabled = autocast(device_type=self.device) if self.device in ["cuda", "cpu"] else nullcontext()

        with torch.no_grad():
            with autocast_enabled:
                analyze_videos(
                    config=self.model_cfg,
                    videos=[video_file],
                    shuffle=self.shuffle,
                    videotype=None,
                    destfolder=out_dir,
                    engine=Engine.PYTORCH,
                    modelprefix=os.path.dirname(self.model_cfg), 
                    device=str(self.device) if self.device in ['cuda', 'mps'] else None, # Passes 'mps' or 'cuda' to DLC
            )
        
            if make_labeled_video:
                vid_stem = Path(video_file).stem
                h5_out_candidates = sorted(
                    (p for p in Path(out_dir).glob(f"{vid_stem}*shuffle{self.shuffle}*.h5")
                        if p.is_file() and not p.stem.endswith("_rodeoflow")),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not h5_out_candidates:
                    raise FileNotFoundError(f"h5 dlc output file for video {vid_stem} not found")
                h5_out = str(h5_out_candidates[0])
                labeled_mp4 = (Path(video_file).parent / f"{Path(h5_out).stem}_labeled.mp4" )
                self.create_labeled_video(
                    video_path=video_file,
                    h5_path=h5_out,
                    pose_cfg_path=str(self.pose_cfg),
                    out_path=str(labeled_mp4),
                )