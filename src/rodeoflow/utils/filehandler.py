from pathlib import Path

def get_files2analyse(keypoint_dir, analysis_type: str, overwrite: bool = False) -> list[Path]:
    '''
    Returns face OR body files found in keypoint_dir that should be analyzed.
    Matches patterns like '*DLC*Mouse{Face|Body}*.h5' and legacy LightningPose files.
    '''
    if analysis_type not in ['face', 'body']:
        raise ValueError(f"Invalid analysis_type: '{analysis_type}'. Must be 'face' or 'body'.")

    if keypoint_dir is None:
        raise RuntimeError("No DLC directory defined.")

    d = Path(keypoint_dir)
    if not d.exists():
        raise FileNotFoundError(f"Keypoint directory does not exist: {d}")

    cased_type = analysis_type.capitalize()  # 'Face' or 'Body'

    if cased_type == 'Body':
        body_files = set(d.glob('*DLC*MouseBody*.h5'))
        body_files.update(d.glob('*LP_MouseBody*.h5'))

        # Recognize older LP files named directly after their video stem
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
        if d.parent.exists():
            for video in d.parent.iterdir():
                if (
                    video.is_file()
                    and video.suffix.lower() in video_extensions
                    and 'labeled' not in video.name.lower()
                ):
                    legacy_file = d / f'{video.stem}.h5'
                    standardized_file = d / f'{video.stem}_LP_MouseBody.h5'
                    if legacy_file.is_file() and not standardized_file.is_file():
                        body_files.add(legacy_file)

        raw_files = sorted(body_files)
    else:
        raw_files = sorted(d.glob(f'*DLC*Mouse{cased_type}*.h5'))

    # Exclude downstream rodeoflow output files
    filtered_files = [f for f in raw_files if not f.name.endswith('_rodeoflow.h5')]

    # Optional: If overwrite=False, exclude input files that already have a corresponding _rodeoflow.h5 output
    if not overwrite:
        filtered_files = [
            f for f in filtered_files 
            if not f.with_name(f"{f.stem}_rodeoflow.h5").exists()
        ]

    return filtered_files