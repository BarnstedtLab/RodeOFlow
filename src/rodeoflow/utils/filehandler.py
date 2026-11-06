from pathlib import Path

def get_files2analyse(keypoint_dir, overwrite, analysis_type):
    '''
    Returns face OR body files which can be found in keypoint_dir and should be analysed.
    The files must match the pattern '*DLC*Mouse{analysis_type}*.h5'.
    '''
    if analysis_type not in ['face', 'body']:
        print("please check configuration. Only analysis_type='face' and analysis_type='body' is supported.")
    if analysis_type == 'face':
        analysis_type = 'Face'
    elif analysis_type == 'body':
        analysis_type = 'Body'
    if keypoint_dir is None:
        raise RuntimeError("No DLC directory defined.")
    
    if analysis_type == 'Body':
        body_files = set(d.glob('*DLC*MouseBody*.h5'))
        body_files.update(d.glob('*LP_MouseBody*.h5'))

        # Also recognize older LP files named exactly after their video.
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
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

        all_files = sorted(body_files)
    else:
        all_files = sorted(d.glob(f'*DLC*Mouse{analysis_type}*.h5'))