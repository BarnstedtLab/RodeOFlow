Quickstart
==========

Run pose estimation first, then analyze the generated keypoint files.

Face analysis
-------------

.. code-block:: python

   from pathlib import Path
   from rodeoflow import pose_face, analyse_face

   video_dir = Path("/path/to/videos")
   keypoint_dir = video_dir / "rodeoflow"

   pose_face(video_dir)
   analyse_face(keypoint_dir)

Body analysis
-------------

.. code-block:: python

   from pathlib import Path
   from rodeoflow import pose_body, analyse_body

   video_dir = Path("/path/to/videos")
   keypoint_dir = video_dir / "rodeoflow"

   pose_body(video_dir)
   analyse_body(keypoint_dir)