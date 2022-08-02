"""Compute subtitle alignment metrics.

This module provides code to evaluate the quality of a set of subtitle alignments (i.e.
how well their start and end times match those of a set of ground-truth subtitle times).

Note: several of the metrics are derived from the following code:
    https://github.com/yabufarha/ms-tcn/blob/
        c6eab71ddd7b4190ffb3bf6f1b57f3517454939b/eval.py#L15

Example usage:
python misc/sub_align/evaluate_sub_alignment.py \
    --pred_subtitle_dir /scratch/shared/beegfs/albanie/shared-datasets/bbcsl_raw/subtitles/subtitles-vtt-text-normalized-aligned/heuristic-aligned-subs-05_01_2021-mouth-padding4_all
"""
import argparse
from pickle import SHORT_BINSTRING
from typing import List, Tuple
from pathlib import Path

import os 
import tqdm
import numpy as np
import webvtt
from beartype import beartype
from config.config import *

opts = load_opts()


@beartype
def get_labels_start_end_time(
        frame_wise_labels: List[int],
        bg_class: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    """Given a single sequence of frame level labels, find: (i) the start index,
    (ii) the end index and (iii) the label, of each contiguous subsequence of labels.

    Args:
        frame_wise_labels: a single sequence of frame-level labels
        bg_class: if given, skip labels that fall within this list of background classes.

    Returns:
        A tuple consisting of three elements:
            the label associated with each subsequence
            the start index associated with each subsequence
            the end index associated with each subsequence
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


@beartype
def f_score(
        recognized: List[int],
        ground_truth: List[int],
        overlap: float,
        bg_class: List[int],
) -> Tuple[float, float, float]:
    """Compute the f-score of a sequence of predicted sequences against a set of ground
    truth annotations (this is the F1 metric used in https://arxiv.org/abs/1903.01945).

    Args:
        recognized: a list of frame-level sequence label predictions
        ground_truth: a list of frame-level sequence label ground truth
        overlap: the F1 overlap threshold
        bg_class: a list of classes that should be excluded from the evaluation

    Returns:
        A tuple containing:
            (i) the total number of true positives
            (i) the total number of false positives
            (i) the total number of false negatives
    """
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x]
                                               for x in range(len(y_label))])

        # Get the best scoring segment
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


@beartype
def subs2frames(
        subs: List[webvtt.Caption],
        max_time: float,
        fps: int,
        exclude_subs: List[int],
        background_label: int,
) -> List[int]:
    """Convert subtitles into a single sequence of frames in which each subtitle is
    assigned a unique integer and the frames covered by that subtitle are assigned this
    integer.

    Args:
        subs: a list of webvtt caption objects, each of which represnts a subtitle caption
           with an associated start and end time as well as text.
        max_time: the maximum duration of the frame sequence to be created (in seconds).
        fps: the frame rate of the videos
        exclude_subs: the indices of subtitles that should be excluded from the evaluation
        background_label: the value to be assigned to frames that are not covered by any
           subtitle (or frames covered by subtitles that are excluded).

    Returns:
        A list of frame-level sequence labels, each indicating the index of the subtitle
        which covered the current frame.
    """
    frames = [background_label for _ in range(round(fps * max_time))]
    for sub_idx, caption in enumerate(subs):
        if sub_idx in exclude_subs:
            continue

        # Ensure that predicted caption alignment time falls within the bounds o fevaluation
        start_time = min(caption._start, max_time)
        end_time = min(caption._end, max_time)

        start_idx = round(fps * start_time)
        end_idx = round(fps * end_time)
        frames[start_idx:end_idx] = [sub_idx for _ in range(end_idx - start_idx)]
    return frames

@beartype
def eval_subtitle_alignment(
        pred_path_root: Path,
        gt_anno_path_root: Path,
        list_videos: List,
        fps: int, 
        shift = 0,
):

    if os.path.exists(os.path.join(gt_anno_path_root, list_videos[0]+'.vtt')): 
        ext_gt = '.vtt'
    else: 
        ext_gt = '/signhd.vtt'

    if os.path.exists(os.path.join(pred_path_root, list_videos[0]+'.vtt')): 
        ext_pred = '.vtt'
    else: 
        ext_pred = '/signhd.vtt'
    gt_anno_paths = [Path(f'{gt_anno_path_root}/{p}{ext_gt}') for p in list_videos]
    pred_paths = [Path(f'{pred_path_root}/{p}{ext_pred}') for p in list_videos]

    """Evaluate subtitle alignment quality.

    Args:
        pred_paths: the locations of subtitle timing predictions (in .vtt format)
        gt_anno_paths: the locations of subtitle ground truth timings (in .vtt format)
        fps: the frame rate of the videos
    """
    correct = 0
    total = 0
    total_subs = 0
    BACKGROUND_LABEL = -1
    MAX_TIME_PAD_SECS = 10
    overlaps = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    for pred_path, gt_path, vid_id in tqdm.tqdm(zip(pred_paths, gt_anno_paths, list_videos)):

        # if 'natural' in str(pred_path):
        #     continue

        pred_subs = list(webvtt.read(pred_path))
        gt_subs = list(webvtt.read(gt_path))

        if shift!=0: 
            for sub_idx in range(len(pred_subs)): 
                pred_subs[sub_idx]._start += shift
                pred_subs[sub_idx]._end += shift

        msg = (f"Expected num. preds {len(pred_subs)} to match num. gt {len(gt_subs)} for"
               f" {pred_path}")

        assert len(pred_subs) == len(gt_subs), msg
        total_subs += len(gt_subs)

        # We pick the maximum time for the evaluation to be a fixed offset (10 seconds)
        # beyond the last ground truth subtitle. The rationale here is that we want to
        # evaluate over the same number of frames for each alignment algorithm (so the
        # evaluation protocol should only depend on the ground truth subtitle alignments)
        # but we also want to penalise methods that predict subtitles beyond the last
        # ground truth alignment.
        max_time = gt_subs[-1]._end + MAX_TIME_PAD_SECS

        # If an annotator is unable to align the subtitle with the signing, they leave
        # a comment in the content of the subtitle itself, which looks like this:
        # "<subtitle-text> [NOT SURE WHERE]"
        # For other subtitles they leave a comment indicating that they do not agree
        # with the interpretation e.g.
        # "<subtitle-text> [INCORRECT]"
        # or that the signing itself is inappropriate/might be offensive
        # "<subtitle-text> [INAPPROPRIATE SIGN]"
        # We exclude these subtitles from the evaluation
        exclude_subs = []
        for sub_idx, sub in enumerate(gt_subs):
            if "[" in sub.text and "]" in sub.text:
                exclude_subs.append(sub_idx)
            # if sub._end - sub._start < 0.01:
            #     exclude_subs.append(sub_idx)
        total_subs -= len(exclude_subs)

        # To compute metrics, we convert all the subtitles into a sequence of frame-level
        # labels, where for each frame, the label indicates which subtitle (if any) it
        # was assigned.
        pred_frames = subs2frames(
            subs=pred_subs,
            max_time=float(max_time),
            exclude_subs=exclude_subs,
            fps=fps,
            background_label=BACKGROUND_LABEL,
        )
        gt_frames = subs2frames(
            subs=gt_subs,
            max_time=float(max_time),
            exclude_subs=exclude_subs,
            fps=fps,
            background_label=BACKGROUND_LABEL,
        )

        # Compute frame-level accuracy
        for pred, gt in zip(pred_frames, gt_frames):
            total += 1
            if pred == gt:
                correct += 1

            msg = (f"Frame-level accuracy: {100 * float(correct)/total:.2f}"
                f" (computed over {total} frames, {total_subs} sentences)")

            # print(msg)
            # import ipdb; ipdb.set_trace(context=20)

        # Compute f-scores at various overlaps over frame sequences
        for ii, overlap in enumerate(overlaps):
            tp1, fp1, fn1 = f_score(
                recognized=pred_frames,
                ground_truth=gt_frames,
                overlap=overlap,
                bg_class=[BACKGROUND_LABEL],
            )
            tp[ii] += tp1
            fp[ii] += fp1
            fn[ii] += fn1

    # Provide a summary of the computed metrics
    print('total ', total, 'subs', total_subs)
    msg = ( f"Computed over {total} frames, {total_subs} sentences - "
            f"Frame-level accuracy: {100 * float(correct)/total:.2f}"
           )
    for ii, overlap in enumerate(overlaps):
        precision = tp[ii] / float(tp[ii] + fp[ii])
        recall = tp[ii] / float(tp[ii] + fn[ii])
        f1 = 2.0 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1) * 100
        f1_msg = (f"F1@{overlap:0.2f}: {f1:.2f}")
        msg = f'{msg} {f1_msg}'

    print(msg)
    return msg 

def parse_args():
    # pylint: disable=line-too-long
    # flake8: noqa: E501
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_subtitle_dir", type=Path, default="/scratch/shared/beegfs/albanie/shared-datasets/bobsl/public_dataset_release/subtitles/audio-aligned")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument( "--gt_anno_dir", type=Path, default="/scratch/shared/beegfs/albanie/shared-datasets/bobsl/public_dataset_release/subtitles/manually-aligned")
    return parser.parse_args()


def main():
    args = parse_args()
    gt_anno_paths = list(args.gt_anno_dir.glob("**/*.vtt"))
    print(f"Found {len(gt_anno_paths)} ground truth annotation files in {args.gt_anno_dir}")
    pred_paths = [args.pred_subtitle_dir / path.relative_to(args.gt_anno_dir) for path
                  in gt_anno_paths]

    #import ipdb; ipdb.set_trace(context=20)

    eval_subtitle_alignment(
        pred_paths=pred_paths,
        fps=args.fps,
        gt_anno_paths=gt_anno_paths,
    )


if __name__ == "__main__":
    main()
