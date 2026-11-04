API reference
=============

Pose estimation
---------------

.. py:function:: pose_face(video_dir, models_dir=None, keypoint_engine_face='DLC', filetype_face='mp4', crop_box_face=None, face_facing='left', keypoint_model_weights_face=None, overwrite=False, save_labled_videos=True, face_videokey=None)

   Detect facial landmarks in videos.

   :param video_dir: Directory containing the input videos.
   :param models_dir: Optional directory for model files.
   :param keypoint_engine_face: Face tracking engine, ``'DLC'`` or ``'LP'``.
   :param filetype_face: Video file extension used to find videos. Defaults to ``'mp4'``.
   :param crop_box_face: Optional crop region for face videos.
   :param face_facing: Face orientation. Defaults to ``'left'``.
   :param keypoint_model_weights_face: Optional path to face model weights.
   :param overwrite: Whether to overwrite existing keypoint outputs.
   :param save_labled_videos: Whether to save videos with the detected landmarks drawn on them.
   :param face_videokey: Optional string used to select videos whose filenames contain that string.

.. py:function:: pose_body(video_dir, models_dir=None, keypoint_engine_body='LP', filetype_body='mp4', crop_box_body=None, body_facing='right', keypoint_model_weights_body=None, overwrite=False, save_labled_videos=True, body_videokey=None)

   Detect body keypoints in videos.

   :param video_dir: Directory containing the input videos.
   :param models_dir: Optional directory for model files.
   :param keypoint_engine_body: Body tracking engine, ``'DLC'`` or ``'LP'``.
   :param filetype_body: Video file extension used to find videos. Defaults to ``'mp4'``.
   :param crop_box_body: Optional crop region for body videos.
   :param body_facing: Body orientation. Defaults to ``'right'``.
   :param keypoint_model_weights_body: Optional path to body model weights.
   :param overwrite: Whether to overwrite existing keypoint outputs.
   :param save_labled_videos: Whether to save videos with the detected keypoints drawn on them.
   :param body_videokey: Optional string used to select videos whose filenames contain that string.

Analysis
--------

.. py:function:: analyse_face(keypoint_dir, overwrite=False, opticalflow_engine='DIS', use_nmf_msks=True, save_vectorfields=False)

   Analyze facial keypoint files and extract facial movement features.

   :param keypoint_dir: Directory containing the face keypoint files.
   :param overwrite: Whether to overwrite existing analysis outputs.
   :param opticalflow_engine: Optical-flow engine: ``'DIS'`` or ``'FB'``.
   :param use_nmf_msks: Whether to use NMF-refined masks for facial regions.
   :param save_vectorfields: Whether to save optical-flow vector fields.

.. py:function:: analyse_body(keypoint_dir, overwrite=False, stride_cfg=None)

   Analyze body keypoint files and extract body movement and stride-cycle features.

   :param keypoint_dir: Directory containing the body keypoint files.
   :param overwrite: Whether to overwrite existing analysis outputs.
   :param stride_cfg: Optional ``StrideConfig`` object for configuring stride-cycle analysis.