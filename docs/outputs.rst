Outputs
=======

RodeOFlow produces pose estimation files and analysis results. Pose estimation
detects landmarks in videos; analysis uses those landmarks to calculate
facial motion or gait features. The available results depend on the tracking
engine, input video, and analysis options.

Output files and folders
------------------------

When ``pose_face`` or ``pose_body`` receives a video directory, it saves
keypoint outputs in a ``rodeoflow`` subdirectory of that directory. The
keypoint file format depends on the tracking engine.

The pose functions can also save videos with detected landmarks drawn on
them. This is controlled by the ``save_labeled_videos`` parameter and is
enabled by default. The current API spells this parameter
``save_labeled_videos``; check the function signature when calling it.

The analysis functions process keypoint files in ``keypoint_dir`` and save
results as HDF5 files with the suffix ``_rodeoflow.h5``. An HDF5 file can
contain multiple tables and datasets. Tables written by pandas can be loaded
by their key; other HDF5 datasets can be read with ``h5py``.

Face analysis outputs
---------------------

``analyse_face`` combines facial landmarks with dense optical flow to produce
facial motion, eye and pupil, spectral, and chewing related outputs. When
``use_nmf_msks=True``, NMF-refined masks are used to define facial regions.

Face HDF5 contents
^^^^^^^^^^^^^^^^^^

Face analysis saves the following items in its HDF5 file:

``face``
    The main face analysis table. It contains framewise measurements and
    derived features, including regional dense optical flow measures, signed motion
    projections, eye and pupil features, and—depending on the analysis—
    spectral and chewing-related features.

``face_anchor``
    An auxiliary table containing face anchor data saved with the analysis.
    Its columns describe the anchor values available for that run.

``facemasks``
    The facial region masks used during analysis. This is stored as an HDF5
    dataset, not as a pandas table.

Regional facial motion
^^^^^^^^^^^^^^^^^^^^^^

Dense optical flow describes image motion between video frames within each
facial region. The ``face`` table can include the following regional signals
for the nose, whisker pad, mouth, and cheek:

* ``MotionEnergy_Nose``, ``MotionEnergy_Whiskerpad``,
  ``MotionEnergy_Mouth``, and ``MotionEnergy_Cheek``: regional motion energy
  signals.
* ``OFmag_Nose``, ``OFmag_Whiskerpad``, ``OFmag_Mouth``, and ``OFmag_Cheek``:
  optical-flow magnitude signals.
* ``OFang_Nose``, ``OFang_Whiskerpad``, ``OFang_Mouth``, and ``OFang_Cheek``:
  optical-flow angle signals.
* ``Projection_Nose``: signed nose region motion projected onto its dominant
  motion axis.
* ``Projection_Whiskerpad``: signed whiskerpad motion projected onto its
  dominant motion axis.

These features describe image motion. They are not direct measurements of
muscle force or displacement in physical units.

Windowed spectral features
^^^^^^^^^^^^^^^^^^^^^^^^^^

The nose and whiskerpad projection signals can be used to calculate
windowed spectral features:

* ``Nose_DominantFrequency_Hz``: frequency of the strongest spectral peak in
  the nose projection, within 2–15 Hz.
* ``Nose_SpectralCentroid_Hz``: energy-weighted center of the nose
  projection's spectrum within 2–15 Hz.
* ``Whiskerpad_DominantFrequency_Hz``: frequency of the strongest spectral
  peak in the whiskerpad projection, within 2–29 Hz.
* ``Whiskerpad_SpectralCentroid_Hz``: energy-weighted center of the
  whiskerpad projection's spectrum within 2–29 Hz.

These estimates are calculated in non-overlapping 3-second windows. The
estimate for each window is assigned to the frames in that window. A dominant
frequency reports the strongest peak in the selected band; if the spectrum
contains multiple peaks, this feature reports only the strongest one. A
spectral centroid summarizes the distribution of spectral energy across the
band; it is not itself a peak frequency. Because the nose and whiskerpad
bands differ, their centroid values should not be treated as directly
comparable.

Windows with more than 20% missing values are skipped. Incomplete trailing
frames that do not fill a full window do not receive a window estimate.

Eye and pupil features
^^^^^^^^^^^^^^^^^^^^^

The ``face`` table can also include eye- and pupil-related measurements.
Depending on the input video and available landmarks, these can include:

* ``PupilDiam``: estimated pupil diameter.
* Pupil-position and eye-opening measurements.
* ``PupilSpeed_EyeOpeningPerSec``: a pupil-speed measure normalized by eye
  opening.
* Blink- and saccade-related measures or event indicators.

These measurements are derived from tracked landmarks. Their quality depends
on landmark tracking and on the visibility of the eye in the video.

Chewing-related features
^^^^^^^^^^^^^^^^^^^^^^^^

Face analysis can include:

* ``Chewing_Envelope``: a signal related to chewing motion.
* ``Chew``: a chewing-related event or feature.

The precise availability of these columns depends on the analysis performed
and the input video.

Body analysis outputs
---------------------

``analyse_body`` calculates framewise body movement and stride cycle gait
features. Its HDF5 file contains the following tables.

``body``
^^^^^^^^

Framewise body measurements calculated from tracked body keypoints. These can
include paw movement and body part angles. Rows correspond to samples from the
analyzed video.

``stride_cycles``
^^^^^^^^^^^^^^^^^

Measurements for detected stride cycles. Each row represents a detected
cycle and can include cycle timing and gait features. Exact columns depend on
the tracked keypoints and stride-analysis configuration.

``processing_summary``
^^^^^^^^^^^^^^^^^^^^^^

A summary of stride detection and processing quality. Use it to review the
processing status and the number and quality of detected strides.

Inspecting and loading HDF5 results
-----------------------------------

Use pandas to list the table keys, dimensions, and columns in an analysis
file:

.. code-block:: python

    import pandas as pd

    file = "path/to/your_file_rodeoflow.h5"

    with pd.HDFStore(file, mode="r") as store:
        print(store.keys())

        for key in store.keys():
            table = store[key]
            print(key, table.shape)
            print(table.columns.tolist())

Load the face tables with pandas:

.. code-block:: python

    face = pd.read_hdf(file, key="face")
    face_anchor = pd.read_hdf(file, key="face_anchor")

Read the ``facemasks`` dataset with ``h5py``:

.. code-block:: python

    import h5py

    with h5py.File(file, mode="r") as h5file:
        facemasks = h5file["facemasks"][...]

Load the body tables by key:

.. code-block:: python

    body = pd.read_hdf(file, key="body")
    strides = pd.read_hdf(file, key="stride_cycles")
    summary = pd.read_hdf(file, key="processing_summary")

Interpreting units and availability
-----------------------------------

Frequency columns ending in ``_Hz`` express frequency in hertz. Landmark
coordinates and optical-flow features are image derived; they should not be
interpreted as distances or speeds in physical units unless the video has
been calibrated and the relevant conversion has been applied.

The columns available in an output file can vary with the video, tracking
engine, and analysis settings. Inspect the HDF5 keys and table columns in the
generated file when determining which results are present.