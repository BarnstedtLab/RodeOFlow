"""
Author: @Jakob Faust (10.10.2026)
Modified by: @Lam Bui (03.11.2026)

Process body keypoint outputs from Lightning Pose or DeepLabCut to extract
framewise body-motion and stride-cycle gait features.
"""


import warnings

import pandas as pd
import numpy as np
from scipy.stats.mstats import zscore

from typing import NamedTuple

from rodeoflow.utils.motion_processing import smooth, freq_analysis2
from rodeoflow.utils.preprocess_video import get_video_for_marker, video_info
from rodeoflow.utils.handling_markers import load_markers
from dataclasses import dataclass
from pathlib import Path
from scipy.signal import savgol_filter, correlate, find_peaks, hilbert, filtfilt, butter, welch

@dataclass(frozen=True)
class StrideConfig:
    body_conf_thresh: float = 0.7
    reference_marker: str = "tail1"
    body_scale_markers: tuple[str, str] | None = ("nose", "tail1")
    smoothing_sec: float = 0.08
    interpolation_sec: float = 0.10
    min_stride_sec: float = 0.12
    max_stride_sec: float = 1.50
    min_prominence_px: float = 8.0
    min_excursion_px: float = 15.0
    min_tracking_fraction: float = 0.90
    min_angle_tracking_fraction: float = 0.70
    x_direction: float = 1.0
    bout_gap_sec: float = 2.0
    min_cycles_per_bout: int = 3

STRIDE_CYCLE_COLUMNS = [
    "limb","stride_id",
    "liftoff_frame",
    "touchdown_frame",
    "next_liftoff_frame",
    "liftoff_time_s",
    "touchdown_time_s",
    "next_liftoff_time_s",
    "stride_duration_s",
    "stride_frequency_hz",
    "swing_duration_s",
    "stance_duration_s",
    "swing_fraction",
    "stance_fraction",
    "duty_factor",
    "peak_relative_time",
    "paw_excursion_px",
    "median_body_length_px",
    "paw_excursion_bodylength",
    "kinematic_intensity_index",
    "liftoff_xrel_px",
    "touchdown_xrel_px",
    "next_liftoff_xrel_px",
    "tracking_fraction",
    "n_internal_troughs",
    "duration_ok",
    "tracking_ok",
    "excursion_ok",
    "touchdown_ok",
    "internal_trough_ok",
    "valid_stride",
    "internal_angle_liftoff_deg",
    "internal_angle_touchdown_deg",
    "internal_angle_next_liftoff_deg",
    "internal_angle_min_deg",
    "internal_angle_max_deg",
    "internal_angle_range_deg",
    "flexion_liftoff_deg",
    "flexion_touchdown_deg",
    "flexion_next_liftoff_deg",
    "flexion_min_deg",
    "flexion_max_deg",
    "flexion_range_deg",
    "max_flexion_phase",
    "angle_tracking_fraction",
    "angle_ok",
    "bout_id",
    "bout_n_cycles",
    "analysis_stride",
]
class AngleResult(NamedTuple):
    angle3: pd.Series
    slope: pd.Series

class PointMotionResult(NamedTuple):
    raw_distance: pd.Series
    proc_distance: pd.Series
    angles: pd.Series

class BodyAnalyzer:
    def __init__(self,cfg: StrideConfig | None = None,overwrite: bool = False,):
        self.cfg = cfg if cfg is not None else StrideConfig()
        self.overwrite = overwrite

    def _as_series(self, x, name, idx):
        x = pd.Series(x, name=name)
        return x.reindex(idx)

    def _dlc_pointdistance(self, point1, point2, body_conf_thresh=.5, body_interpolation_limit=300):
        point1, point2 = point1.values, point2.values
        confident = [(point1[:, 2] > body_conf_thresh), (point2[:, 2] > body_conf_thresh)]
        point1_conf = pd.DataFrame(np.array((point1[:,0]*confident[0], point1[:,1]*confident[0])).T, columns = ['x', 'y'])
        point1_conf[point1_conf == 0] = np.nan
        point2_conf = pd.DataFrame(np.array((point2[:,0]*confident[1], point2[:,1]*confident[1])).T, columns = ['x', 'y'])
        point2_conf[point2_conf == 0] = np.nan
        xydiff_raw = point1_conf - point2_conf
        xydist_raw = pd.Series(np.linalg.norm(xydiff_raw, axis=1))
        xydist_interp = xydist_raw.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        xydist_z = ((xydist_interp - xydist_interp.mean()) / xydist_interp.std(ddof=0))[:len(point1)]

        return pd.Series(xydist_raw), pd.Series(xydist_z)

    def _dlc_angle(self, point1, point2, point3, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150) -> AngleResult:
        point1, point2, point3 = point1.values, point2.values, point3.values
        confident = [(point1[:,2] > body_conf_thresh), (point2[:,2] > body_conf_thresh), (point3[:,2] > body_conf_thresh)]
        point1_conf = pd.DataFrame(np.array((point1[:,0]*confident[0], point1[:,1]*confident[0])).T, columns = ['x', 'y'])
        point1_conf[point1_conf == 0] = np.nan
        point2_conf = pd.DataFrame(np.array((point2[:,0]*confident[1], point2[:,1]*confident[1])).T, columns = ['x', 'y'])
        point2_conf[point2_conf == 0] = np.nan
        point3_conf = pd.DataFrame(np.array((point3[:,0]*confident[2], point3[:,1]*confident[2])).T, columns = ['x', 'y'])
        point3_conf[point3_conf == 0] = np.nan
        dX1 = np.array(point2_conf['x'] - point1_conf['x'], dtype=float)
        dY1 = np.array(point2_conf['y'] - point1_conf['y'], dtype=float)
        deg1 = pd.Series(np.arctan2(dY1, dX1)) + np.pi
        deg1.loc[deg1>np.pi] = deg1.loc[deg1>np.pi] - 2*np.pi
        dX2 = np.array(point3_conf['x'] - point2_conf['x'], dtype=float)
        dY2 = np.array(point3_conf['y'] - point2_conf['y'], dtype=float)
        deg2 = pd.Series(np.arctan2(dY2, dX2))  + np.pi
        deg2.loc[deg2>np.pi] = deg2.loc[deg2>np.pi] - 2*np.pi
        angle = deg2 - deg1
        deg1_interp = deg1.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        angle_interp = angle.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        deg1_smooth = pd.Series(smooth(deg1_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
        angle_smooth = pd.Series(smooth(angle_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
        return AngleResult(angle3=pd.Series(angle_smooth), slope=pd.Series(deg1_smooth))

    def _dlc_pointmotion(self, dlc, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150) -> PointMotionResult:
        x = dlc['x']
        y = dlc['y']
        likelihood = dlc['likelihood']
        confident = likelihood > body_conf_thresh
        x_conf = x*confident
        y_conf = y*confident
        xy = pd.DataFrame(np.array((x_conf, y_conf)).T, columns=['x', 'y'])
        xy[xy == 0] = np.nan
        xydiff_raw = xy.diff()
        xydist_raw = pd.Series(np.linalg.norm(xydiff_raw, axis=1))
        deg = pd.Series(np.arctan2(xydiff_raw.y, xydiff_raw.x)) + np.pi
        deg.loc[deg>np.pi] = deg.loc[deg>np.pi] - 2*np.pi
        xydist_interp = xydist_raw.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        deg_interp = deg.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        xydist_z_raw = ((xydist_interp - xydist_interp.mean()) / xydist_interp.std(ddof=0))[:len(x)]
        xydist_smooth = pd.Series(smooth(xydist_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
        xydist_z = ((xydist_smooth - xydist_smooth.mean()) / xydist_smooth.std(ddof=0))[:len(x)]

        return PointMotionResult(pd.Series(xydist_raw), pd.Series(xydist_z), pd.Series(deg_interp))

    def _clean_coord(self, point, coord, fps):
        interp_limit = max(1, int(round(self.cfg.interpolation_sec * fps)))
        window_len = max(3, int(np.ceil(self.cfg.smoothing_sec * fps)))
        if window_len % 2 == 0:
            window_len += 1
        values = point[coord].astype(float).where(point["likelihood"].astype(float) >= self.cfg.body_conf_thresh)
        values = values.interpolate(limit=interp_limit,limit_direction="both",limit_area="inside",)
        valid = values.notna()
        temporary = values.interpolate(limit_direction="both").ffill().bfill()
        if temporary.isna().all():
            return pd.Series(np.nan, index=point.index, dtype=float)
        smoothed = smooth(temporary.to_numpy(), window_len=window_len)
        extra = len(smoothed) - len(temporary)
        left = extra // 2
        smoothed = smoothed[left:left + len(temporary)]
        cleaned = pd.Series(smoothed, index=point.index, dtype=float)
        cleaned.loc[~valid] = np.nan
        return cleaned

    import warnings
    def _body_length(self, markers, fps):
        marker_pair = self.cfg.body_scale_markers
        if marker_pair is None:
            return pd.Series(np.nan, index=markers.index, dtype=float)
        available = set(markers.columns.get_level_values(0))
        missing = [name for name in marker_pair if name not in available]
        if missing:
            warnings.warn(f"Cannot calculate body-length normalization; "f"missing marker(s): {missing}. Keeping pixel-based features.",RuntimeWarning,stacklevel=2,)
            return pd.Series(np.nan, index=markers.index, dtype=float)
        marker_a, marker_b = marker_pair
        x_a = self._clean_coord(markers[marker_a], "x", fps)
        y_a = self._clean_coord(markers[marker_a], "y", fps)
        x_b = self._clean_coord(markers[marker_b], "x", fps)
        y_b = self._clean_coord(markers[marker_b], "y", fps)
        body_length = pd.Series(np.hypot(x_a - x_b, y_a - y_b),index=markers.index,dtype=float,)
        return body_length.where(body_length > 0)

    def _internal_angle(self, markers, points, fps):
        point1, point2, point3 = points
        x1 = self._clean_coord(markers[point1], "x", fps)
        y1 = self._clean_coord(markers[point1], "y", fps)
        x2 = self._clean_coord(markers[point2], "x", fps)
        y2 = self._clean_coord(markers[point2], "y", fps)
        x3 = self._clean_coord(markers[point3], "x", fps)
        y3 = self._clean_coord(markers[point3], "y", fps)
        vector1 = np.column_stack([x1 - x2, y1 - y2])
        vector2 = np.column_stack([x3 - x2, y3 - y2])
        denominator = (np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2, axis=1))
        cosine = np.divide(np.sum(vector1 * vector2, axis=1),denominator,out=np.full(len(markers), np.nan),where=denominator > 0,)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        angle = pd.Series(angle, index=markers.index)
        flexion = 180 - angle
        return angle, flexion

    def _extract_strides(self, paw_xrel, body_length, internal_angle, flexion, fps):
        # Fill gaps temporarily for event detection. The exported trace retains them.
        interp_limit = max(1, int(round(self.cfg.interpolation_sec * fps)))
        detect = paw_xrel.interpolate(limit=interp_limit,limit_direction="both",limit_area="inside",)
        empty = pd.DataFrame()
        if detect.notna().sum() < 3:
            return empty, 0, 0, 0
        filled = detect.interpolate(limit_direction="both").ffill().bfill()
        if filled.isna().any():
            return empty, 0, 0, 0
        troughs, _ = find_peaks(-filled.to_numpy(dtype=float),prominence=self.cfg.min_prominence_px,distance=max(1, int(round(self.cfg.min_stride_sec * fps))),)
        n_candidates = max(0, len(troughs) - 1)
        rows = []
        for stride_id, (start, stop) in enumerate(zip(troughs[:-1], troughs[1:])):
            start, stop = int(start), int(stop)
            duration_s = (stop - start) / fps
            raw_segment = paw_xrel.iloc[start:stop + 1]
            segment = filled.iloc[start:stop + 1].to_numpy(dtype=float)
            if len(segment) < 3:
                continue
            excursion_px = float(np.max(segment) - np.min(segment))
            internal_troughs, _ = find_peaks(-segment[1:-1],prominence=max(3.0, 0.15 * excursion_px),distance=max(1, int(np.ceil(0.05 * fps))),)
            touchdown_local = int(np.argmax(segment))
            touchdown = start + touchdown_local
            swing_s = (touchdown - start) / fps
            stance_s = (stop - touchdown) / fps
            tracking_fraction = float(raw_segment.notna().mean())
            body_segment = body_length.iloc[start:stop + 1]
            median_body_length_px = (float(body_segment.median()) if body_segment.notna().any() else np.nan)
            excursion_bodylength = (excursion_px / median_body_length_px if np.isfinite(median_body_length_px) and median_body_length_px > 0 else np.nan)
            duration_ok = (self.cfg.min_stride_sec <= duration_s <= self.cfg.max_stride_sec)
            tracking_ok = (tracking_fraction >= self.cfg.min_tracking_fraction)
            excursion_ok = excursion_px >= self.cfg.min_excursion_px
            touchdown_ok = start < touchdown < stop
            internal_trough_ok = len(internal_troughs) == 0
            angle_segment = internal_angle.iloc[start:stop + 1]
            flexion_segment = flexion.iloc[start:stop + 1]
            angle_values = angle_segment.to_numpy(dtype=float)
            flexion_values = flexion_segment.to_numpy(dtype=float)
            angle_tracking_fraction = float(angle_segment.notna().mean())
            angle_ok = angle_tracking_fraction >= self.cfg.min_angle_tracking_fraction
            internal_angle_min = (float(np.nanmin(angle_values)) if np.isfinite(angle_values).any() else np.nan)
            internal_angle_max = (float(np.nanmax(angle_values)) if np.isfinite(angle_values).any() else np.nan)
            flexion_min = (float(np.nanmin(flexion_values)) if np.isfinite(flexion_values).any() else np.nan)
            flexion_max = (float(np.nanmax(flexion_values))if np.isfinite(flexion_values).any() else np.nan)
            max_flexion_local = (int(np.nanargmax(flexion_values)) if np.isfinite(flexion_values).any() else -1)
            max_flexion_phase = (max_flexion_local / (len(flexion_values) - 1) if max_flexion_local >= 0 and len(flexion_values) > 1 else np.nan)
            rows.append({
                "stride_id": stride_id,
                "liftoff_frame": start,
                "touchdown_frame": touchdown,
                "next_liftoff_frame": stop,
                "stride_duration_s": duration_s,
                "stride_frequency_hz": 1.0 / duration_s,
                "swing_duration_s": swing_s,
                "stance_duration_s": stance_s,
                "swing_fraction": swing_s / duration_s,
                "stance_fraction": stance_s / duration_s,
                "duty_factor": stance_s / duration_s,
                "paw_excursion_px": excursion_px,
                "median_body_length_px": median_body_length_px,
                "paw_excursion_bodylength": excursion_bodylength,
                "liftoff_xrel_px": float(segment[0]),
                "touchdown_xrel_px": float(segment[touchdown_local]),
                "next_liftoff_xrel_px": float(segment[-1]),
                "tracking_fraction": tracking_fraction,
                "n_internal_troughs": len(internal_troughs),
                "duration_ok": duration_ok,
                "tracking_ok": tracking_ok,
                "excursion_ok": excursion_ok,
                "touchdown_ok": touchdown_ok,
                "internal_trough_ok": internal_trough_ok,
                "valid_stride": (duration_ok and tracking_ok and excursion_ok and touchdown_ok and internal_trough_ok),
                "internal_angle_liftoff_deg": internal_angle.iloc[start] if angle_ok else np.nan,
                "internal_angle_touchdown_deg": internal_angle.iloc[touchdown] if angle_ok else np.nan,
                "internal_angle_next_liftoff_deg": internal_angle.iloc[stop] if angle_ok else np.nan,
                "internal_angle_min_deg": internal_angle_min if angle_ok else np.nan,
                "internal_angle_max_deg": internal_angle_max if angle_ok else np.nan,
                "internal_angle_range_deg": internal_angle_max - internal_angle_min if angle_ok else np.nan,
                "flexion_liftoff_deg": flexion.iloc[start] if angle_ok else np.nan,
                "flexion_touchdown_deg": flexion.iloc[touchdown] if angle_ok else np.nan,
                "flexion_next_liftoff_deg": flexion.iloc[stop] if angle_ok else np.nan,
                "flexion_min_deg": flexion_min if angle_ok else np.nan,
                "flexion_max_deg": flexion_max if angle_ok else np.nan,
                "flexion_range_deg": flexion_max - flexion_min if angle_ok else np.nan,
                "max_flexion_phase": max_flexion_phase if angle_ok else np.nan,
                "angle_tracking_fraction": angle_tracking_fraction,
                "angle_ok": angle_ok,
                "peak_relative_time": swing_s / duration_s,
                "kinematic_intensity_index": (excursion_bodylength / duration_s if np.isfinite(excursion_bodylength) and duration_s > 0 else np.nan)
            })

        stride_table = pd.DataFrame(rows)
        stride_table["bout_id"] = pd.Series(pd.NA, index=stride_table.index, dtype="Int64")
        stride_table["bout_n_cycles"] = 0
        stride_table["analysis_stride"] = False
        if not stride_table.empty:
            eligible = stride_table.loc[stride_table["valid_stride"] & stride_table["swing_fraction"].between(0, 1)].copy()
            if not eligible.empty:
                gap_s = (eligible["liftoff_frame"] - eligible["next_liftoff_frame"].shift()) / fps
                eligible["bout_id"] = (gap_s.fillna(np.inf) > self.cfg.bout_gap_sec).cumsum() - 1
                eligible["bout_n_cycles"] = eligible.groupby("bout_id")["stride_id"].transform("size")
                stride_table.loc[eligible.index, "bout_id"] = (eligible["bout_id"].astype("Int64"))
                stride_table.loc[eligible.index, "bout_n_cycles"] = (eligible["bout_n_cycles"])
                stride_table.loc[eligible.index, "analysis_stride"] = (eligible["bout_n_cycles"] >= self.cfg.min_cycles_per_bout)
        n_qc_valid = (int(stride_table["valid_stride"].sum()) if not stride_table.empty else 0)
        n_internal_rejected = (int((~stride_table["internal_trough_ok"]).sum()) if not stride_table.empty else 0)
        return stride_table, n_candidates, n_qc_valid, n_internal_rejected


    def analyse(self, body_file: Path, keypoint_dir: Path):
        out_file = body_file.with_name(body_file.stem + '_rodeoflow.h5')
        if out_file.exists() and not self.overwrite:
            print(f"{out_file} already analysed, skipping ahead...")
            return

        markers_body = load_markers(body_file, key=None)
        video_file = get_video_for_marker(body_file, keypoint_dir)
        fps, w, h, cap = video_info(video_file)

        # Paw motion
        dlc_paw_front = (markers_body.loc[:, ('paw_front-right2', ['x','y','likelihood'])].droplevel(0, axis=1))
        dlc_paw_back = ( markers_body.loc[:, ('paw_back-right2', ['x','y','likelihood'])].droplevel(0, axis=1))
        motion_frontpaw = self._dlc_pointmotion(dlc_paw_front)
        motion_backpaw = self._dlc_pointmotion(dlc_paw_back)
        # Paw angles
        angle_paws_front = self._dlc_angle(markers_body['paw_front-right1'], markers_body['paw_front-right2'], markers_body['paw_front-right3'])
        angle_paws_back = self._dlc_angle( markers_body['paw_back-right1'],  markers_body['paw_back-right2'],  markers_body['paw_back-right3'])
        # Stride and gait information
        rightpaws_fbdiff_raw, _  = self._dlc_pointdistance(markers_body['paw_front-right2'], markers_body['paw_back-right2'])
        stride_freq = freq_analysis2(rightpaws_fbdiff_raw, fps)
        # Mouth motion
        dlc_mouth = (markers_body.loc[:, ('mouth', ['x','y','likelihood'])].droplevel(0, axis=1))
        motion_mouth = self._dlc_pointmotion(dlc_mouth)
        # Tail information
        angle_tail = self._dlc_angle(markers_body['tail1'], markers_body['tail2'], markers_body['tail3'])
        idx = markers_body.index

        # Therefore, we just fill up the shorter rows with NaNs to match the raw data lengths.
        body_raw = pd.concat([
            self._as_series(motion_frontpaw.raw_distance, 'PointMotion_FrontPaw', idx),
            self._as_series(motion_frontpaw.angles,       'AngleMotion_FrontPaw', idx),
            self._as_series(motion_mouth.raw_distance,    'PointMotion_Mouth', idx),
            self._as_series(motion_mouth.angles,          'AngleMotion_Mouth', idx),
            self._as_series(motion_backpaw.raw_distance,  'PointMotion_BackPaw', idx),
            self._as_series(motion_backpaw.angles,        'AngleMotion_BackPaw', idx),
            self._as_series(stride_freq, 'Frequency_BackPaw', idx),
            self._as_series(angle_tail.angle3,            'Angle_Tail_3', idx),
            self._as_series(angle_tail.slope,             'Angle_Tail', idx),
            self._as_series(angle_paws_front.angle3,      'Angle_Paws_Front_3', idx),
            self._as_series(angle_paws_front.slope,       'Angle_Paws_Front', idx),
            self._as_series(angle_paws_back.angle3,       'Angle_Paws_Back_3', idx),
            self._as_series(angle_paws_back.slope,        'Angle_Paws_Back', idx),
            self._as_series(-zscore(markers_body['tail1','y']), 'Tailroot_Level', idx),
        ], axis=1)

        body_length = self._body_length(markers_body, fps)

        stride_tables = []
        processing_rows = []
        limb_specs = [
            ("fore","paw_front-right2",("paw_front-right1", "paw_front-right2", "paw_front-right3"),),
            ("hind","paw_back-right2",("paw_back-right1", "paw_back-right2", "paw_back-right3"),
            ),]
        ref_x = self._clean_coord(markers_body[self.cfg.reference_marker], "x", fps)
        for limb, paw_name, angle_points in limb_specs:
            paw_x = self._clean_coord(markers_body[paw_name], "x", fps)
            paw_xrel = self.cfg.x_direction * (paw_x - ref_x)
            internal_angle, flexion = self._internal_angle(markers_body, angle_points, fps)

            prefix = "ForePaw" if limb == "fore" else "HindPaw"
            body_raw[f"{prefix}_XRel_px"] = paw_xrel
            body_raw[f"{prefix}_InternalAngle_deg"] = internal_angle
            body_raw[f"{prefix}_Flexion_deg"] = flexion

            cycles, n_candidates, n_qc_valid, n_internal_rejected = (self._extract_strides(paw_xrel, body_length, internal_angle, flexion, fps))
            processing_rows.append({
                "limb": limb,
                "fps": fps,
                "candidate_cycles": n_candidates,
                "qc_valid_cycles": n_qc_valid,
                "internal_trough_rejections": n_internal_rejected,
                "stride_rows_saved": len(cycles),
                "analysis_strides": (int(cycles["analysis_stride"].sum())if not cycles.empty else 0),
            })
            if not cycles.empty:
                cycles.insert(0, "limb", limb)
                cycles["liftoff_time_s"] = cycles["liftoff_frame"] / fps
                cycles["touchdown_time_s"] = cycles["touchdown_frame"] / fps
                cycles["next_liftoff_time_s"] = cycles["next_liftoff_frame"] / fps
                stride_tables.append(cycles)
        stride_cycles = (pd.concat(stride_tables, ignore_index=True) if stride_tables else pd.DataFrame())
        stride_cycles = stride_cycles.reindex(columns=STRIDE_CYCLE_COLUMNS)
        processing_summary = pd.DataFrame(processing_rows)
        if "bout_id" in stride_cycles.columns: stride_cycles["bout_id"] = stride_cycles["bout_id"].astype("float64")
        stride_cycles.to_hdf(out_file, key="stride_cycles", mode="a")
        processing_summary.to_hdf(out_file, key="processing_summary", mode="a")
        # further process raw data and save
        cap.release()
        body_raw.to_hdf(out_file, key='body')
        print(f"saved rodeoflow body analysis to {out_file}")
