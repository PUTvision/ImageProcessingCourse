import json
import platform
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Tuple, Dict, Any
import os
import time

import click
import numpy as np
import pandas as pd
# from sklearn import metrics


@click.command()
@click.argument('applications_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('input_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('output_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('result_file_gt', type=click.Path(exists=True, dir_okay=False))
@click.option('--no-run', is_flag=True)
@click.option('--no-compute', is_flag=True)
def main(
        applications_directory: str,
        input_directory: str,
        output_directory: str,
        result_file_gt: str,
        no_run: bool,
        no_compute: bool
) -> None:
    applications_directory = Path(applications_directory)
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    result_file_gt = Path(result_file_gt)

    if not no_run:
        states = run_applications(applications_directory, input_directory, output_directory)
        with open(output_directory / 'states.json', 'w') as states_file:
            json.dump(states, states_file)

    if not no_compute:
        results = compute_results(output_directory, result_file_gt)
        with open(output_directory / 'results.json', 'w') as results_file:
            json.dump(results, results_file)
        df_results = pd.DataFrame.from_dict(results, orient='index')
        df_results.sort_index(inplace=True)
        df_results.to_csv(output_directory / 'results.csv', index=True)


def run_applications(
        applications_directory: Path,
        input_file: Path,
        output_directory: Path
) -> Dict[str, str]:
    states = {}
    for applications_directory_entry in applications_directory.iterdir():
        if applications_directory_entry.is_dir():
            student_name, status = process_application_directory(
                applications_directory_entry, input_file, output_directory
            )
            states[student_name] = status

    return states


def compute_results(output_directory: Path, result_file_gt: Path) -> Dict[str, float]:
    """
    Computes scores for result files in subdirectories of output_directory against a ground truth file.

    The score for each processed directory is the sum of absolute differences for each metric
    within corresponding video segments. Every video segment (top-level key) present in the
    ground truth is considered required.

    If a video segment or a specific metric within a segment is present in the ground truth
    but missing from the processed file, its value in the processed file is treated as 0
    for the purpose of calculating the absolute difference.

    Args:
        output_directory (Path): The path to the directory containing subdirectories,
                                 each expected to hold a result JSON file.
        result_file_gt (Path): The path to the ground truth JSON file.

    Returns:
        Dict[str, float]: A dictionary where keys are the names of the processed directories
                          and values are their calculated scores (sum of absolute differences).
                          Returns an empty dictionary if the ground truth file cannot be loaded
                          or no valid result files are found.
    """
    scores: Dict[str, float] = {}

    # Load the ground truth data from the specified path.
    try:
        with open(result_file_gt, 'r', encoding='utf-8') as f:
            ground_truth_data: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{result_file_gt}'. Please ensure the path is correct.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from ground truth file '{result_file_gt}'. "
              "Please check if the file contains valid JSON.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading ground truth file '{result_file_gt}': {e}")
        return {}

    # Iterate through all entries within the specified output_directory.
    for subdir in output_directory.iterdir():
        # Process only if the entry is a directory.
        if subdir.is_dir():
            subdir_name = subdir.name
            print(f"Processing directory: '{subdir_name}'")

            # Assumption: The result file inside each subdirectory is named 'results.json'.
            # If your result files have a different name, you will need to change 'results.json' below.
            result_file_path = subdir / "results.json"

            # Check if the assumed result file exists in the current subdirectory.
            if not result_file_path.is_file():
                print(f"  Warning: Result file expected at '{result_file_path}' not found. Skipping '{subdir_name}'.")
                # If the result file is missing, we still need to calculate a score,
                # assuming all processed values are 0.
                processed_data = {} # Treat as empty
            else:
                processed_data: Dict[str, Any] = {}
                # Load the processed data from the result file in the current subdirectory.
                try:
                    with open(result_file_path, 'r', encoding='utf-8') as f:
                        processed_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"  Error: Could not decode JSON from '{result_file_path}'. "
                          "Treating all processed values as 0 for this directory.")
                    processed_data = {} # Treat as empty for scoring
                except Exception as e:
                    print(f"  An unexpected error occurred while reading '{result_file_path}': {e}. "
                          "Treating all processed values as 0 for this directory.")
                    processed_data = {} # Treat as empty for scoring


            total_score_for_subdir = 0.0

            # Compare ground truth data with the processed data.
            # We iterate through ALL video segments present in the ground truth, as they are all required.
            for video_segment_key, gt_metrics in ground_truth_data.items():
                # Get the metrics for this video segment from the processed data.
                # If the video segment key is missing in processed_data, this will default to an empty dict.
                # This ensures that for missing segments, all metrics are treated as 0 in processed_value.
                processed_metrics = processed_data.get(video_segment_key, {})

                # Iterate through the metrics for this video segment from the ground truth.
                for metric_key, gt_value in gt_metrics.items():
                    # Get the processed value for the current metric.
                    # If the metric key is missing in processed_metrics, default to 0.
                    processed_value = processed_metrics.get(metric_key, 0)

                    # Calculate the absolute difference for the current metric.
                    # Ensure values are numeric for subtraction.
                    try:
                        diff = abs(float(gt_value) - float(processed_value))
                        total_score_for_subdir += diff
                        # Optional: print details of differences
                        # if diff > 0:
                        #     print(f"    Difference found in '{video_segment_key}' for '{metric_key}': "
                        #           f"GT={gt_value}, Processed={processed_value}, Diff={diff}")
                    except (ValueError, TypeError):
                        print(f"    Warning: Non-numeric values encountered for '{metric_key}' in "
                              f"'{video_segment_key}'. GT: {gt_value}, Processed: {processed_value}. "
                              "Skipping this metric for score calculation.")
                        continue

            # Store the calculated total score (sum of absolute differences) for the current subdirectory.
            scores[subdir_name] = total_score_for_subdir
            print(f"  Total score for '{subdir_name}': {total_score_for_subdir} (sum of absolute differences).")

    return scores


def compute_results_old(output_directory: Path, result_file_gt: Path) -> Dict[str, float]:
    global_results = {}

    print(result_file_gt)
    with result_file_gt.open() as file_gt:
        VALID_RESULTS = json.load(file_gt)
    print(VALID_RESULTS)

    for student_output_directory in output_directory.iterdir():
        if not student_output_directory.is_dir():
            continue

        results_file_path: Path = student_output_directory / 'results.json'
        try:
            with results_file_path.open() as results_file:
                results = json.load(results_file)

            print(results)
            score = 0
            for image_name, valid_result in VALID_RESULTS.items():
                image_result = results[image_name] if image_name in results else 0
                # image_result = image_result if len(image_result) == len(valid_result) else 0

                single_image_score = 0
                for char_valid, char_image in zip(valid_result, image_result):
                    if char_valid == char_image:
                        single_image_score += 1

                if single_image_score == len(valid_result) and len(image_result) == len(valid_result):
                    single_image_score += 3

                score += single_image_score
            global_results[student_output_directory.name] = score
        except Exception as e:
            print(f'{student_output_directory.name} failed: {e}', file=sys.stderr)

    return global_results


def process_application_directory(
        path: Path,
        input_file: Path,
        output_dir: Path
) -> Tuple[str, str]:
    print(f'Processing "{path.name}"...')
    requirements_file = path / 'requirements.txt'
    if requirements_file.exists():
        print('Installing external dependencies...')
        temp_venv_dir = Path(tempfile.gettempdir()) / 'SiSWVenv'
        venv_builder = venv.EnvBuilder(system_site_packages=True, clear=True, with_pip=True)
        venv_builder.create(str(temp_venv_dir))
        is_windows = any(platform.win32_ver())
        interpreter = str(temp_venv_dir / 'scripts' / 'python') if is_windows else str(temp_venv_dir / 'bin' / 'python')
        try:
            subprocess.check_call([interpreter, '-m', 'pip', '-qqq', 'install', '-r', str(requirements_file)])
        except subprocess.CalledProcessError:
            return path.name, 'PIPFAILED'
    else:
        interpreter = sys.executable

    for application_file in path.iterdir():
        if application_file.name.endswith('.py'):
            student_name = application_file.name[:-3]
            student_output_dir = output_dir / student_name
            student_output_dir.mkdir(exist_ok=True)
            results_file = student_output_dir / 'results.json'
            stdout_file = student_output_dir / 'stdout'
            stderr_file = student_output_dir / 'stderr'

            print(f'Running "{student_name}"...')
            try:
                with stdout_file.open('w') as stdout, stderr_file.open('w') as stderr:
                    start_time = time.time()
                    subprocess.run(
                        [interpreter, os.path.abspath(str(application_file)), os.path.abspath(str(input_file))+'\\', os.path.abspath(str(results_file))],
                        cwd=str(path), stdout=stdout, stderr=stderr, timeout=((6*60+50)*30*0.05 + 600)  # 6 minutes 50 seconds, 30 fps, 0.05 seconds per frame, plus 60s grace period
                    )
                    print("--- %s seconds ---" % (time.time() - start_time))
            except subprocess.TimeoutExpired:
                return student_name, 'TIMEOUT'
            except subprocess.SubprocessError:
                return student_name, 'ERROR'

            return student_name, 'OK'

    return path.name, 'NOSCRIPT'


if __name__ == '__main__':
    main()
