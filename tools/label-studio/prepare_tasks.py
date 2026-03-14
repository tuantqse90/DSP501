"""
Prepare Label Studio tasks from UrbanSound8K dataset.
Selects representative samples per class for manual frequency analysis.
"""
import os
import sys
import json
import shutil
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config

import pandas as pd


def prepare_tasks(samples_per_class: int = 5, output_dir: str = None):
    """
    Select representative samples and create Label Studio import file.

    Args:
        samples_per_class: Number of samples to select per class.
        output_dir: Directory for output files.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # Load metadata
    df = pd.read_csv(config.METADATA_PATH)

    # Create audio serving directory
    audio_dir = os.path.join(output_dir, 'audio_samples')
    os.makedirs(audio_dir, exist_ok=True)

    random.seed(config.RANDOM_SEED)
    tasks = []

    for cls_id, cls_name in enumerate(config.CLASS_NAMES):
        subset = df[df['classID'] == cls_id]

        # Prefer foreground (salience=1) samples
        foreground = subset[subset['salience'] == 1]
        if len(foreground) >= samples_per_class:
            selected = foreground.sample(n=samples_per_class, random_state=config.RANDOM_SEED)
        else:
            selected = subset.sample(n=min(samples_per_class, len(subset)), random_state=config.RANDOM_SEED)

        for _, row in selected.iterrows():
            src = os.path.join(config.AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
            dst = os.path.join(audio_dir, row['slice_file_name'])

            if os.path.exists(src):
                shutil.copy2(src, dst)
                tasks.append({
                    'data': {
                        'audio': f"/data/local-files/?d=audio_samples/{row['slice_file_name']}",
                    },
                    'meta': {
                        'class': cls_name,
                        'classID': int(cls_id),
                        'fold': int(row['fold']),
                        'fsID': int(row['fsID']),
                        'salience': int(row['salience']),
                        'filename': row['slice_file_name'],
                    }
                })

    # Save tasks JSON
    tasks_file = os.path.join(output_dir, 'tasks.json')
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"Prepared {len(tasks)} tasks ({samples_per_class} per class)")
    print(f"Audio files copied to: {audio_dir}")
    print(f"Tasks file: {tasks_file}")
    print(f"\nClasses included:")
    for cls_name in config.CLASS_NAMES:
        count = sum(1 for t in tasks if t['meta']['class'] == cls_name)
        print(f"  {cls_name}: {count} samples")

    return tasks_file


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5, help='Samples per class')
    args = parser.parse_args()
    prepare_tasks(samples_per_class=args.samples)
