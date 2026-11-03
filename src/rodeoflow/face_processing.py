"""
Author: @Jakob Faust (10.10.2026)
Modified by: @Lam Bui (03.11.2026)

Analyze facial videos using keypoint-derived masks and dense optical flow.
Extract pupil and eyelid dynamics, blinks and saccades, nose and whiskerpad
motion and frequency features, and chewing-related signals.
"""

from pathlib import Path
from scipy import signal
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
import h5py
from dataclasses import dataclass
from scipy.ndimage import median_filter


from rodeoflow.utils import confidence_na, load_markers
from rodeoflow.optical_flow import BaseOF, FarnebackOF, DISFlowOF, DynamicMasks
from rodeoflow.utils.preprocess_video import get_video_for_marker, video_info
from rodeoflow.utils.motion_processing import smooth, freq_analysis2, dlc_pointdistance2, hilbert_peaks

@dataclass(frozen=True)
class FaceConfig:
    conf_thresh: float | None = None
    interpolation_limits_sec: dict = ...
    smoothing_windows_sec: dict = ...
    na_limit: float = 0.25
    faceregions_sizes: dict | None = None
    base_resolution: tuple[int,int] | None = None
    manual_anchor: dict | None = None

class FaceAnalyzer():
    def __init__(self, cfg:FaceConfig=None, opticalflow_enigne:str='DIS', save_vectorfield: bool=False):
        self.opticalflow_enigne = opticalflow_enigne
        self.save_vectorfield = save_vectorfield

        if cfg:
            self.cfg = cfg
        else:
            conf_thresh = 0.5
            interpolation_limits_sec={
                'pupil': 2,
                'eyelid': 1,
            }
            smoothing_windows_sec={
                'PupilDiam': 1,
                'PupilMotion': 0.25,
                'eyelid': 0.1,
                'MotionEnergy': 0.25,
            }
            na_limit=0.25
            faceregions_sizes=None
            base_resolution=None
            manual_anchor=None
            self.cfg = FaceConfig(conf_thresh, interpolation_limits_sec,
                       smoothing_windows_sec, na_limit, faceregions_sizes,
                       base_resolution, manual_anchor)

    def _process_raw_data(self, smoothing_windows_sec, na_limit, FaceCam_FPS, interpolation_limits_frames, face_raw):
        face_raw.iloc[:, (face_raw.isnull().mean()>na_limit).values] = np.nan

        # Interpolate missing values
        face_interp = face_raw.copy()
        face_interp[['PupilX', 'PupilY', 'PupilMotion', 'PupilDiam']] = \
                face_interp[['PupilX', 'PupilY', 'PupilMotion', 'PupilDiam']].interpolate(
                    method='linear', limit=interpolation_limits_frames['pupil'])

        # Smoothen data
        smoothing_windows_frames = {x: int(k * FaceCam_FPS)
                                    for (x, k) in smoothing_windows_sec.items()}
        face_smooth = face_interp.copy()
        face_smooth['PupilDiam'] = face_smooth['PupilDiam'].rolling(
                window=smoothing_windows_frames['PupilDiam'], center=True).mean().astype('float32')
        face_smooth[['PupilX', 'PupilY', 'PupilMotion']] = \
                face_smooth[['PupilX', 'PupilY', 'PupilMotion']].rolling(
                window=smoothing_windows_frames['PupilMotion'], center=True).mean().astype('float32')
        face_smooth.loc[:, face_smooth.columns.str.startswith('MotionEnergy')] = \
                face_smooth.loc[:, face_smooth.columns.str.startswith('MotionEnergy')].rolling(
                window=smoothing_windows_frames['MotionEnergy'], center=True).mean().astype('float32')

        # Detect saccades from 2D pupil speed
        pupil_x = median_filter(face_smooth["PupilX"].to_numpy(dtype=float), size=3)
        pupil_y = median_filter(face_smooth["PupilY"].to_numpy(dtype=float), size=3)
        vx = np.diff(pupil_x) * FaceCam_FPS
        vy = np.diff(pupil_y) * FaceCam_FPS
        speed_px_s = np.hypot(vx, vy)
        # Estimate a fixed open-eye reference from the larger valid eyelid
        # separations in this recording. Do not normalize frame by frame:
        # eyelid separation changes during blinks.
        lid_dist = face_interp["EyeLidDist"].to_numpy(dtype=float)
        valid_lid_dist = lid_dist[np.isfinite(lid_dist) & (lid_dist > 0)]
        if valid_lid_dist.size:
            upper_lid_cutoff = np.quantile(valid_lid_dist, 0.80)
            reference_lid_separation = np.median(valid_lid_dist[valid_lid_dist >= upper_lid_cutoff])
        else:
            reference_lid_separation = np.nan
        if np.isfinite(reference_lid_separation) and reference_lid_separation > 0:
            speed_normalized = speed_px_s / reference_lid_separation
        else:
            speed_normalized = np.full(speed_px_s.shape, np.nan)
        # Align transition speeds to the frame where each transition ends.
        speed_by_frame = np.full(len(face_smooth), np.nan)
        speed_by_frame[1:] = speed_normalized
        face_smooth["PupilSpeed_EyeOpeningPerSec"] = pd.Series(speed_by_frame, index=face_smooth.index)
        # Z-score continuous signals. The normalized speed is also retained
        # in face_smooth in its interpretable normalized units.
        face_zscore = face_smooth.apply(lambda a: (a - a.mean()) / a.std(ddof=0))
        # Adaptive saccade thresholds with hysteresis, applied to normalized speed.
        is_saccade = np.zeros(len(speed_normalized), dtype=bool)
        valid_speed = np.isfinite(speed_normalized)
        if valid_speed.any():
            mu = np.nanmean(speed_normalized)
            sigma = np.nanstd(speed_normalized)
            high_thr = mu + 1.5 * sigma
            low_thr = mu + 1.0 * sigma
            candidate_frames = np.flatnonzero(speed_normalized > high_thr)
            for peak in candidate_frames:
                start = peak
                while start > 0 and speed_normalized[start] > low_thr:
                    start -= 1
                end = peak
                while (end < len(speed_normalized) - 1 and speed_normalized[end] > low_thr):
                    end += 1
                is_saccade[start:end + 1] = True
            is_saccade &= valid_speed
        saccade_frames = np.zeros(len(face_smooth), dtype=int)
        saccade_frames[1:] = is_saccade.astype(int)
        face_zscore["Saccades"] = pd.Series(saccade_frames, index=face_smooth.index)
        face_zscore["EyeBlink"] = pd.Series(face_interp["EyeLidDist"] < face_interp["EyeLidDist"].median() * 0.75,index=face_smooth.index,).astype(int)
        # Concatenating all data types into multi-level dataframe
        face = pd.concat({'raw': face_raw, 'interpolated': face_interp, 'smooth': face_smooth, 'zscore': face_zscore}, names=['Data_type'], axis=1)

        return face

    def _pupilextraction(self, pupil_markers_xy_confident):
        pupil_circle = np.zeros(
            shape=(len(pupil_markers_xy_confident), 2), dtype=object)
        print("Fitting circle to pupil...")

        # 2D points from DLC MouseFace
        for i in tqdm(range(len(pupil_markers_xy_confident))):
            pupilpoints = np.float32(pupil_markers_xy_confident[i].reshape(6, 2))
            pupil_circle[i, :] = cv2.minEnclosingCircle(pupilpoints)

        # extract pupil centroid
        pupil_centre = np.asarray(tuple(pupil_circle[:, 0]), dtype=np.float32)
        pupil_x_raw = pd.Series(pupil_centre[:, 0])
        pupil_y_raw = pd.Series(pupil_centre[:, 1])

        # extract pupil movement
        pupil_xy = pd.DataFrame(
            np.array((pupil_x_raw, pupil_y_raw)).T, columns=['x', 'y'])
        pupil_xy[pupil_xy == 0] = np.nan
        pupil_xydist_raw = pd.Series(np.linalg.norm(pupil_xy.diff(), axis=1))

        # extract pupil diameter
        pupil_diam_raw = pd.Series(pupil_circle[:, 1], dtype=np.float32)

        return pd.DataFrame({'PupilX': pupil_x_raw,
                            'PupilY': pupil_y_raw,
                            'PupilMotion': pupil_xydist_raw,
                            'PupilDiam': pupil_diam_raw})

    def _eyeblink(self, dlc_face, eyelid_conf_thresh=.999, eyelid_smooth_window=25, eyelid_interpolation_limit=150):
        eyelid_labels_top = np.array(dlc_face.values[:, [15, 16]]) * (dlc_face.values[:, [17, 17]] > eyelid_conf_thresh)
        eyelid_labels_bot = np.array(dlc_face.values[:, [18, 19]]) * (dlc_face.values[:, [20, 20]] > eyelid_conf_thresh)
        eyelid_labels_top[eyelid_labels_top == 0] = np.nan
        eyelid_labels_bot[eyelid_labels_bot == 0] = np.nan
        eyelid_dist_raw = pd.Series(np.linalg.norm(eyelid_labels_top - eyelid_labels_bot, axis=1))
        eyelid_dist_interp = eyelid_dist_raw.interpolate(method='linear', limit=eyelid_interpolation_limit)  # linear interpolation, 2sec maximum
        eyelid_dist_smooth = pd.Series(smooth(eyelid_dist_interp, window_len=eyelid_smooth_window)).shift(periods=-int(eyelid_smooth_window/2))
        eyelid_dist_z = (eyelid_dist_smooth - eyelid_dist_smooth.mean()) / eyelid_dist_smooth.std(ddof=0)
        eyeblinks = eyelid_dist_interp < eyelid_dist_interp.median()*.75

        return eyelid_dist_z, eyeblinks

    # FACE REGIONS
    def _define_faceregions(self, dlc_face, facevid, dlc_file, manual_anchor=None, faceregions_sizes=None, base_resolution=(782, 582)):
        # checking on video
        facemp4 = cv2.VideoCapture(facevid)
        if not facemp4.isOpened():
            raise IOError(f"Cannot open video file: {facevid}")
        ret, frame = facemp4.read()
        facemp4.release()
        if not ret or frame is None:
            raise IOError(f"Cannot read first frame from: {facevid}")
        firstframe = frame[:, :, 0].astype(np.uint8)
        plt.imshow(firstframe, cmap='gray');

        # create empty canvas
        canvas = np.zeros(firstframe.shape, dtype=np.uint8)

        # define face anchors
        anchor_names = ['nosetip', 'forehead', 'mouthtip',
                        'chin', 'tearduct', 'eyelid2']
        face_anchor = pd.DataFrame(index=['x', 'y'], columns=anchor_names)
        for anchor_name in anchor_names:
            if manual_anchor and anchor_name in manual_anchor:
                x, y = manual_anchor[anchor_name]
                face_anchor.at['x', anchor_name] = x
                face_anchor.at['y', anchor_name] = y
                plt.scatter(x, y, c='yellow', s=40)
            else:
                if dlc_face[anchor_name, 'x'].isnull().mean() == 1:  # if only missing values, leave empty
                    face_anchor.at['x', anchor_name] = np.nan
                    face_anchor.at['y', anchor_name] = np.nan
                else:
                    x = dlc_face[anchor_name, 'x'].mean()
                    y = dlc_face[anchor_name, 'y'].mean()
                    face_anchor.at['x', anchor_name] = x
                    face_anchor.at['y', anchor_name] = y
                    plt.scatter(dlc_face[anchor_name]['x'],
                                dlc_face[anchor_name]['y'], alpha=.002)
                    plt.scatter(face_anchor[anchor_name]['x'],
                                face_anchor[anchor_name]['y'])
        face_anchor['eyelid_bottom'] = face_anchor['eyelid2'] # in DLC the keypoints are labeled with 1,2,3,etc.,
        face_anchor = face_anchor.astype(float)               # ensure numeric

        # compute scaling if sizes provided
        H, W = firstframe.shape
        if base_resolution is None:
            base_resolution = (H, W)
        w0, h0 = base_resolution
        scale_x = W / w0
        scale_y = H / h0

        def get_axes(region):
            """
            If provide manual faceregions_sizes, this returns the
            axes (width, height) for the given region in pixels, scaled
            from the base_resolution to the current video resolution.
            Otherwise returns None to trigger the automatic size.
            """
            if faceregions_sizes and region in faceregions_sizes:
                sx0, sy0 = faceregions_sizes[region]
                # scale from base_resolution to actual frame size
                return (sx0 * scale_x, sy0 * scale_y)
            return None

        def create_mask(centre, axes, angle):
            sx, sy = map(int, axes)
            cv2.ellipse(firstframe, centre, (sx, sy), angle, 0, 360, (255,0,0), 4)
            mask = cv2.ellipse(canvas.copy(), centre, (sx, sy), angle, 0, 360, (1,), -1)
            return mask.astype(bool)

        def get_pt(name):
            arr = face_anchor.loc[['x','y'], name].values
            return None if np.isnan(arr).all() else arr

        nose_pt    = get_pt('nosetip')
        mouth_pt   = get_pt('mouthtip')
        tear_pt    = get_pt('tearduct')
        forehead_pt= get_pt('forehead')
        chin_pt    = get_pt('chin')
        eyelid_pt  = get_pt('eyelid_bottom')
        face_anchor.at['x', 'whiskerpad_center'] = np.nan
        face_anchor.at['y', 'whiskerpad_center'] = np.nan

        # whisker inference
        whisker_r = 0.0
        if nose_pt is None or mouth_pt is None or tear_pt is None:
            mask_whiskers = canvas.astype(bool)
        else:
            centre_whiskers = tuple(np.round(np.vstack([
                nose_pt,
                mouth_pt,
                tear_pt]).mean(axis=0)).astype(int))
            face_anchor.at['x', 'whiskerpad_center'] = centre_whiskers[0]
            face_anchor.at['y', 'whiskerpad_center'] = centre_whiskers[1]
            a = np.linalg.norm(mouth_pt - tear_pt)
            b = np.linalg.norm(tear_pt  - nose_pt)
            c = np.linalg.norm(nose_pt  - mouth_pt)
            s = 0.5*(a+b+c)
            whisker_r = math.sqrt(s*(s-a)*(s-b)*(s-c)) / s
            axes = get_axes('whiskers') or (whisker_r * 1.1, whisker_r * 1.1)
            mask_whiskers = create_mask(centre_whiskers, axes, angle=0)
        # nose inference
        if nose_pt is None:
            mask_nose = canvas.copy()
        else:
            centre_nose = tuple((nose_pt + np.array([0.24 * nose_pt[0], 0.12 * nose_pt[1]])).round().astype(int))
            axes = get_axes('nose') or (whisker_r*2/3, whisker_r*1/2)
            mask_nose = create_mask(centre_nose, axes, angle=-60.0)
        # mouth inference ellipse
        if mouth_pt is None or chin_pt is None:
            mask_mouth = canvas.astype(bool)
        else:
            centre_mouth = tuple(np.round(mouth_pt + (chin_pt - mouth_pt)/3).astype(int))
            angle_mouth = math.degrees(math.atan2(chin_pt[1]  - mouth_pt[1], chin_pt[0]  - mouth_pt[0]))
            dist = np.hypot(chin_pt[0]  - mouth_pt[0], chin_pt[1]  - mouth_pt[1])
            axes = get_axes('mouth') or (dist / 1.75, dist / 4)
            mask_mouth = create_mask(centre_mouth, axes, angle_mouth)
        # cheek inference ellipse
        if chin_pt is None or eyelid_pt is None:
            mask_cheek = canvas.astype(bool)
        else:
            centre_cheek = tuple(np.round(eyelid_pt + (chin_pt - eyelid_pt)/2).astype(int))
            dist = np.linalg.norm(chin_pt - eyelid_pt)
            axes = get_axes('cheek') or ((2.5/5)*dist, (1/4)*dist)
            mask_cheek = create_mask(centre_cheek, axes, angle=0)
        masks = [mask_nose, mask_whiskers, mask_mouth, mask_cheek]
        if dlc_file:
            plt.imshow(firstframe, cmap='gray')
            plt.savefig(str(dlc_file)[:-3] + "_face_regions.png")
        plt.close('all')
        return masks, face_anchor

    def _has_gpu_support(self):
        '''
        Tests if the installation has proper GPU support (and if a GPU is actually available).
        First check: OpenCV with CUDA support (needs to be compiled on machine!!)
        Second check: PyTorch with CUDA support

        if 'Farneback' is specified as of_backend and there is no opencv CUDA installation,
        this will still return False even if it runs on a GPU system!!
        '''
        has_cv2_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

        if not has_cv2_cuda and (self.opticalflow_enigne == "FB"):
            print("No Opencv with GPU support detected, sorry. Ensure to have a working nvidia GPU." \
                "If you have, try switching mf_config.of_backend to 'RAFT' ")
            return False

        if has_cv2_cuda and (self.opticalflow_enigne == "FB"):
            return True

        return False # safe fallback

    def _facemotion(self, videopath, masks, face_anchor, videoslice=[], use_nmf_msks=True):
        cap = cv2.VideoCapture(str(videopath))
        if videoslice:
            cap.set(cv2.CAP_PROP_POS_FRAMES, videoslice[0])
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read video {videopath} for DynamicMasks")

        height, width = frame.shape[:2]
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cap.release()

        # Generate NMF masks
        if use_nmf_msks:
            print("Generating dynamic masks via NMF...")
            dyn_masks = DynamicMasks(str(videopath), width, height)
            dyn_masks.set_masks(masks)
            dyn_masks.set_anchors(face_anchor)
            masks_sp, px_counts = dyn_masks.generate_updated_masks(prev_gray)
            
            # Convert sparse masks back to a list of dense 2D arrays 
            dense_masks_flat = masks_sp.toarray()
            updated_masks = []
            for m in dense_masks_flat:
                # Reshape and ensure mask matches (height, width) orientation
                mask_2d = m.reshape((height, width), order='C').astype(np.float32)
                updated_masks.append(mask_2d)
            masks = updated_masks # Overwrite the original masks
        else:
            print("Continuing without NMF masks")

        # Initialize and run the Optical Flow
        gpu_flow : BaseOF = None
        if self.opticalflow_enigne == 'FB':
            gpu_flow = FarnebackOF()
        elif self.opticalflow_enigne == 'DIS':
            gpu_flow = DISFlowOF()
        else:
            raise RuntimeError(f"Unexpected flow type {self.opticalflow_enigne}, expected 'DIS' or 'FB'.")

        gpu_flow.save_vectors = self.save_vectorfield

        if videoslice:
            start, end = videoslice[0], videoslice[1]
        else:
            start, end = 0, None

        gpu_flow.open(videopath, start=start, end=end)
        gpu_flow.set_masks(masks)
        out = gpu_flow.run()

        motion = np.hstack([out['diff'], out['mag'], out['ang']])
        cols = ['MotionEnergy_Nose','MotionEnergy_Whiskerpad','MotionEnergy_Mouth','MotionEnergy_Cheek',
                'OFmag_Nose','OFmag_Whiskerpad','OFmag_Mouth','OFmag_Cheek',
                'OFang_Nose','OFang_Whiskerpad','OFang_Mouth','OFang_Cheek']

        df_statistics = pd.DataFrame(motion, columns=cols)
        flow_grid = out["flow_grid"] if self.save_vectorfield else None

        return df_statistics, flow_grid

    def _to_h5(self, out_file: Path, face_masks, face_anchor: pd.DataFrame, face: pd.DataFrame):
            mf_out_file = str(out_file)
            with h5py.File(mf_out_file, "a") as out:
                if "facemasks" in out:
                    del out["facemasks"]
                out.create_dataset("facemasks", data=face_masks)
            face_anchor.to_hdf(mf_out_file, key="face_anchor", mode="a")
            face.to_hdf(mf_out_file, key="face", mode="a")
            print(f"saved rodeoflow face analysis to {mf_out_file}")

    def _interpolation_limits(self, fps: float, min_frames: int=1):
        out : dict[str, int] = {}
        for key, val in self.cfg.interpolation_limits_sec.items():
            if val is None or (isinstance(val, (int, float)) and val <= 0):
                out[key] = min_frames
            else:
                out[key] = int(max(min_frames, round(float(val) * float(fps))))
        return out

    def _perform_pca(self, angles, magnitudes, axis=None, reference_vector=None):
        u = magnitudes * np.cos(angles)
        v = magnitudes * np.sin(angles)
        u = u.flatten()
        v = v.flatten()
        if (np.isnan(u).any() or np.isnan(v).any()):
            print(f"found nans: u -> {np.isnan(u).sum()}, v -> {np.isnan(v).sum()}")
        u = np.nan_to_num(u, 0.)
        v = np.nan_to_num(v, 0.)

        X = np.column_stack((u, v))
        pca = PCA(n_components=1)
        pca.fit(X)

        if axis is None:
            main_axis = pca.components_[0]
            # Flip axis if it points away from the reference vector
            if reference_vector is not None:
                if np.dot(main_axis, reference_vector) < 0:
                    main_axis[0] *= -1
                    main_axis[1] *= -1
                    print('Main axis rotated to align with reference vector!')
        else:
            main_axis = axis

        explained_var = pca.explained_variance_ratio_[0]

        print(f"Main Axis of Motion: u={main_axis[0]:.2f}, v={main_axis[1]:.2f}")
        print(f"This axis explains {explained_var*100:.1f}% of all movement.")

        # Project data onto the final axis
        projection = u * main_axis[0] + v * main_axis[1]
        return projection, main_axis
    @staticmethod
    def _windowed_spectral_metric(x,fps,metric,fmin,fmax,window_sec=3.0,):
        """Estimate a dominant spectral peak or spectral centroid per window."""
        if metric not in {"peak", "centroid"}:
            raise ValueError("metric must be 'peak' or 'centroid'")
        if fps <= 0:
            raise ValueError("fps must be positive")
        series = x if isinstance(x, pd.Series) else pd.Series(x)
        values = series.to_numpy(dtype=float)
        window_len = int(round(window_sec * fps))
        output = np.full(len(values), np.nan)
        if window_len < 3:
            return pd.Series(output, index=series.index)
        for start in range(0, len(values) - window_len + 1, window_len):
            stop = start + window_len
            segment = values[start:stop]
            finite = np.isfinite(segment)
            # Skip windows with too much missing tracking data.
            if finite.mean() < 0.8:
                continue
            segment = (pd.Series(segment).interpolate(limit_direction="both").to_numpy(dtype=float))
            nfft = max(1024, 1 << (len(segment) - 1).bit_length())
            frequencies, power = signal.welch(segment,fs=fps,window="hann",nperseg=len(segment),noverlap=0,nfft=nfft,detrend="constant",scaling="density",)
            band = (frequencies >= fmin) & (frequencies <= fmax)
            band_freq = frequencies[band]
            band_power = power[band]
            if len(band_freq) == 0 or not np.isfinite(band_power).all():
                continue
            if metric == "peak":
                estimate = band_freq[np.argmax(band_power)]
            else:
                total_power = np.trapz(band_power, band_freq)
                if total_power <= 0:
                    continue
                estimate = np.trapz(band_freq * band_power, band_freq) / total_power
            output[start:stop] = estimate
        return pd.Series(output, index=series.index)


    @staticmethod
    def _hilbert_peaks(x, fps, fc=None, butterN=5, peakprom=1.5):
        """Return the filtered Hilbert envelope and binary mouth-event peaks."""
        series = x if isinstance(x, pd.Series) else pd.Series(x)
        values = series.to_numpy(dtype=float)
        env = np.abs(signal.hilbert(values))
        nyquist = fps / 2.0
        fc_used = nyquist * 0.5 if fc is None or fc >= nyquist else fc
        b, a = signal.butter(butterN, fc_used / nyquist, "low")
        shift_amt = int(0.6 * fps)
        envelope = pd.Series(signal.filtfilt(b, a, env),index=series.index,).shift(shift_amt)
        peaks, _ = signal.find_peaks(envelope.to_numpy(), prominence=peakprom)
        peak_flags = np.zeros(len(values), dtype=int)
        peak_flags[peaks] = 1

        return envelope, pd.Series(peak_flags, index=series.index)

    def analyse(self, face_file: Path, landmark_dir: Path, use_nmf_msks=True):
        out_file = face_file.with_name(face_file.stem + '_rodeoflow.h5')
        out_grid = face_file.with_name(face_file.stem + '_optical_flow_grid.npz')
        if out_file.exists() and not getattr(self, 'overwrite', False):
            print(f"{out_file} already analysed, skipping ahead...")
            return
        markers_face = load_markers(face_file, key=None)
        video_file = get_video_for_marker(face_file, landmark_dir)
        fps, w, h, cap = video_info(video_file)
        markers_face = confidence_na(self.cfg.conf_thresh, markers_face)
        interpolation_limits = self._interpolation_limits(fps)

        markers_face.loc[:, ['pupil'+str(n+1) for n in range(6)]] = \
            markers_face.loc[:, ['pupil'+str(n+1) for n in range(6)]].interpolate(
                method='linear', limit=interpolation_limits['pupil'])

        # Extracting pupil and eyelid data
        pupil_raw = self._pupilextraction(
            markers_face[['pupil'+str(n+1) for n in range(6)]].values)
        eyelid_dist_raw = pd.Series(dlc_pointdistance2(
            markers_face['eyelid1'], markers_face['eyelid2']), name='EyeLidDist')

        # Define and save face regions
        face_masks, face_anchor = self._define_faceregions(
            markers_face, video_file, face_file, manual_anchor=self.cfg.manual_anchor,
            base_resolution=self.cfg.base_resolution, faceregions_sizes=self.cfg.faceregions_sizes
        )

        # Extract motion in face regions
        if not self._has_gpu_support():
            self.opticalflow_enigne = 'DIS'
            print("No CUDA support detected. Processing with DIS optical flow on your CPU...")
        face_motion, flow_grid = self._facemotion(
            video_file, face_masks, face_anchor, use_nmf_msks=use_nmf_msks)

        # Perform PCA on Nose (Reference: Downwards, positive Y)
        nose_pca_proj, _ = self._perform_pca(
            face_motion['OFang_Nose'].values,
            face_motion['OFmag_Nose'].values,
            reference_vector=np.array([0.0, 1.0])
        )
        face_motion['Projection_Nose'] = nose_pca_proj

        # Perform PCA on Whiskerpad (Reference: Towards Nosetip)
        nose_pt = face_anchor.loc[['x', 'y'], 'nosetip'].values
        whisker_pt = face_anchor.loc[['x', 'y'], 'whiskerpad_center'].values

        if not np.isnan(nose_pt.astype(float)).any() and not np.isnan(whisker_pt.astype(float)).any():
            ref_whisker = nose_pt - whisker_pt
        else:
            ref_whisker = np.array([1.0, 0.0]) # Fallback if points are missing

        whisker_pca_proj, _ = self._perform_pca(
            face_motion['OFang_Whiskerpad'].values,
            face_motion['OFmag_Whiskerpad'].values,
            reference_vector=ref_whisker
        )
        face_motion['Projection_Whiskerpad'] = whisker_pca_proj

        nose_freq = self._windowed_spectral_metric(face_motion["Projection_Nose"],fps,metric="peak",fmin=2,fmax=15,)
        whiskerpad_centroid = self._windowed_spectral_metric(face_motion["Projection_Whiskerpad"],fps,metric="centroid",fmin=2,fmax=29,)
        chewenv, chew = self._hilbert_peaks(face_motion["OFang_Mouth"],fps,)
        face_freq = pd.DataFrame({"Nose_DominantFrequency_Hz": nose_freq,"Whiskerpad_SpectralCentroid_Hz": whiskerpad_centroid, "Chewing_Envelope": chewenv,"Chew": chew,})
        face_raw = pd.concat([pupil_raw, eyelid_dist_raw, face_motion, face_freq], axis=1)

        cap.release()
        face = self._process_raw_data(self.cfg.smoothing_windows_sec, self.cfg.na_limit, fps, interpolation_limits, face_raw)
        self._to_h5(out_file, face_masks, face_anchor, face)
        if flow_grid is not None:
            np.savez_compressed(out_grid, flow=flow_grid)
