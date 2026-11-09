from pathlib import Path

def get_files2analyse(keypoint_dir, overwrite, analysis_type):
    """Return tracking H5 files that still need face or body analysis."""
    if analysis_type not in {"face", "body"}:
        raise ValueError("analysis_type must be 'face' or 'body'.")
    if keypoint_dir is None:
        raise ValueError("A keypoint directory is required.")

    d = Path(keypoint_dir)
    if not d.is_dir():
        raise NotADirectoryError(d)

    model_name = "MouseFace" if analysis_type == "face" else "MouseBody"
    candidates = set(d.glob(f"*DLC*{model_name}*.h5"))

    if analysis_type == "body":
        candidates.update(d.glob("*_LP_MouseBody.h5"))

        # Support older LP predictions named after the source video.
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
        for video in d.parent.iterdir():
            if (
                video.is_file()
                and video.suffix.lower() in video_extensions
                and "labeled" not in video.stem.lower()
            ):
                legacy = d / f"{video.stem}.h5"
                standardized = d / f"{video.stem}_LP_MouseBody.h5"
                if legacy.is_file() and not standardized.is_file():
                    candidates.add(legacy)

    files_to_analyze = []
    for path in sorted(candidates):
        if not path.is_file() or path.stem.endswith("_rodeoflow"):
            continue
        output = path.with_name(f"{path.stem}_rodeoflow.h5")
        if overwrite or not output.exists():
            files_to_analyze.append(path)

    return files_to_analyze
