Outputs
=======

RodeOFlow creates pose-estimation files and analysis files. The available
features depend on the tracking engine, video, and analysis options.

Output files
------------

When a pose-estimation function receives a video directory, it saves keypoint
outputs in a ``rodeoflow`` subdirectory. The file format can depend on the
tracking engine.

The analysis functions save results as HDF5 files with the suffix
``_rodeoflow.h5``. HDF5 files can contain one or more tables; the table names
are called *keys*.

Face-analysis outputs
---------------------

Face analysis combines facial landmarks with dense optical flow to extract
framewise measurements of facial movement.

Regional motion
^^^^^^^^^^^^^^^

Optical-flow measurements describe image motion within facial regions. Depending
on the analysis, the output includes:

* Regional motion-energy signals (``MotionEnergy_*``).
* Optical-flow magnitude signals (``OFmag_*``).
* Optical-flow angle signals (``OFang_*``).
* Signed motion projections for the nose (``Projection_Nose``) and whisker pad
  (``Projection_Whiskerpad``).

These signals describe movement in the video. They are not direct measurements
of muscle force or movement in physical units.

Eye and pupil features
^^^^^^^^^^^^^^^^^^^^^^

Face analysis also extracts pupil and eye-related measurements, including
pupil diameter and position, eye opening, and pupil movement. The output can
include blink and saccade-related measures.

Frequency and chewing features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The analysis can include frequency-related features derived from the nasal and
whisker-pad motion signals, such as ``Nose_DominantFrequency_Hz`` and
``Whiskerpad_SpectralCentroid_Hz``. Chewing-related outputs include
``Chewing_Envelope`` and ``Chew``.

Body-analysis outputs
---------------------

Body analysis extracts framewise body movement features and measurements of
detected stride cycles. Its HDF5 files contain these tables:

* ``body``: framewise body measurements, including body-part movement and
  angles.
* ``stride_cycles``: measurements for detected stride cycles, including
  stride timing and gait features.
* ``processing_summary``: a summary of stride detection and processing quality.

Inspecting an HDF5 file
-----------------------

Use pandas to list the available tables in an analysis file:

.. code-block:: python

   import pandas as pd

   file = "path/to/your_file_rodeoflow.h5"

   with pd.HDFStore(file, mode="r") as store:
       print(store.keys())

For a body-analysis file, load the tables by name:

.. code-block:: python

   body = pd.read_hdf(file, key="body")
   strides = pd.read_hdf(file, key="stride_cycles")
   summary = pd.read_hdf(file, key="processing_summary")

The exact face-analysis table keys and column names should be checked in the
generated file with ``store.keys()``. Column availability can also depend on
the analysis options and the input video.