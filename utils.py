import numpy as np
import os

import torch
from typing import List, Tuple
from tqdm import tqdm
from datetime import datetime, timedelta
import webvtt
import pickle 
import matplotlib.pyplot as plt
from jiwer import wer as wer_calc
# -------------------- Colorize ------------------------------------------
"""A set of common utilities used within the environments. These are
not intended as API functions, and will not remain stable over time.
"""
import numpy as np
import matplotlib.colors as colors
from sklearn.metrics import precision_recall_curve

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

color2num = dict(gray=30,
                 red=31,
                 green=32,
                 yellow=33,
                 blue=34,
                 magenta=35,
                 cyan=36,
                 white=37,
                 crimson=38)

def plot_confidences(
    probs, 
    times_gt,
    times_ref,
    times_neg, 
    b_id, 
    dir_output,
):
    dir_output = f"{dir_output}/plts_10w_0.5sec_logits_forwardonly/"
    os.makedirs(dir_output, exist_ok=True)
    file_path = f"{dir_output}/{b_id}.png"
    plt.axvline(x=times_gt[0], c='r')
    plt.axvline(x=times_gt[1], c='r')
    plt.plot(times_ref, [probs[0],probs[0]], 'c', linewidth=5)
    
    for idx, w in enumerate(times_neg):
        plt.plot(w, [probs[idx], probs[idx]], c='k')
    plt.savefig(file_path)
    plt.close()

def plot_nce_score_nb_annots(
    scores_gt,
    annot_words_all,
    annot_scores_all,
    annot_types_all,
    text_gt_all,
    times_gt_all,
    dir_output,
):
    scores_gt = np.asarray(scores_gt)
    annot_words_all = np.asarray(annot_words_all)
    annot_scores_all = np.asarray(annot_scores_all)
    annot_types_all = np.asarray(annot_types_all)
    text_gt_all = np.asarray(text_gt_all)
    times_gt_all = np.asarray(times_gt_all)
    dir_output = f"{dir_output}/scores_vs_nb_annots/"
    os.makedirs(dir_output, exist_ok=True)
    # file_path = f"{dir_output}/scores_vs_nb_annots.png"
    np.save(f"{dir_output}/scores_gt.npy",scores_gt)
    np.save(f"{dir_output}/annot_words_all.npy",annot_words_all)
    np.save(f"{dir_output}/annot_scores_all.npy",annot_scores_all)
    np.save(f"{dir_output}/annot_types_all.npy",annot_types_all)
    np.save(f"{dir_output}/text_gt_all.npy",text_gt_all)
    np.save(f"{dir_output}/times_gt_all.npy",times_gt_all)
    # plt.scatter(nb_annots, scores_gt)
    # plt.xlabel('Nb spottings (mouthings/spottings)')
    # plt.ylabel('Score from model') 
    # plt.savefig(file_path)
    # plt.close()


def plot_precision_recall(
    scores_all,
    iou_2_labels,
    dir_output,
    score_type,
):
    dir_output = f"{dir_output}/precision_recall_curves/"
    os.makedirs(dir_output, exist_ok=True)
    file_path = f"{dir_output}/precision_recall_{score_type}.png"
    for overlap in iou_2_labels.keys():
        lr_precision, lr_recall, _ = precision_recall_curve(iou_2_labels[overlap], scores_all)
        plt.plot(lr_recall, lr_precision, label=f"IOU {overlap}")
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.legend(loc='lower right')
    plt.savefig(file_path)
    plt.close()  

def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    # Import six here so that `utils` has no import-time dependencies.
    # We want this since we use `utils` during our import-time sanity checks
    # that verify that our dependencies (including six) are actually present.
    import six

    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(six.u(str(num)))
    if bold:
        attr.append(six.u('1'))
    attrs = six.u(';').join(attr)
    return six.u('\x1b[%sm%s\x1b[0m') % (attrs, string)


def calc_iou(times_gt, time):
    a_s, a_e = times_gt
    b_s, b_e = time
    if b_s > a_e or a_s > b_e:
        return 0
    else:
        o_s = max(a_s,b_s)
        o_e = min(a_e,b_e)
        intersection = o_e - o_s
        u_s = min(a_s,b_s)
        u_e = max(a_e,b_e)
        union = u_e - u_s
        return intersection/float(union) 

def green(s):
    return colorize(s, 'green', bold=True)


def blue(s):
    return colorize(s, 'blue', bold=True)


def red(s):
    return colorize(s, 'red', bold=True)


def magenta(s):
    return colorize(s, 'magenta', bold=True)


def colorize_mat(mat, hsv):
    """
    Colorizes the values in a 2D matrix MAT
    to the color as defined by the color HSV.
    The values in the matrix modulate the 'V' (or value) channel.
    H,S (hue and saturation) are held fixed.

    HSV values are assumed to be in range [0,1].

    Returns an uint8 'RGB' image.
    """
    mat = mat.astype(np.float32)
    m, M = np.min(mat), np.max(mat)
    v = (mat - m) / (M - m)
    h, s = hsv[0] * np.ones_like(v), hsv[1] * np.ones_like(v)
    hsv = np.dstack([h, s, v])
    rgb = (255 * colors.hsv_to_rgb(hsv)).astype(np.uint8)
    return rgb


# -------------------- / Colorize ------------------------------------------


def gpu_initializer(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    return device


def calc_1d_iou(bb1, bb2):
    """
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : (x1,x2)
    bb2 : (x1,x2)

    Returns
    -------
    float
        in [0, 1]
    """
    # assert bb1[0] < bb1[1]
    # assert bb2[0] < bb2[1]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    x_right = min(bb1[1], bb2[1])

    if x_right < x_left:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) 

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) 
    bb2_area = (bb2[1] - bb2[0]) 

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou



def calc_1d_iou_batch(bb1, bb2):
    """
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : batch_size x (x1,x2)
    bb2 : batch_size x (x1,x2)

    Returns
    -------
    float
        in [0, 1]
    """

    #bb1[:,0] = torch.minimum(bb1[:,0],bb1[:,1])
    #bb2[:,0] = torch.minimum(bb2[:,0],bb2[:,1])

    # determine the coordinates of the intersection rectangle
    x_left = torch.maximum(bb1[:,0], bb2[:,0])
    x_right = torch.minimum(bb1[:,1], bb2[:,1])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left)*(x_right - x_left>0)

    # compute the area of both AABBs
    bb1_area = (bb1[:,1] - bb1[:,0])*(bb1[:,1] - bb1[:,0]>0)
    bb2_area = (bb2[:,1] - bb2[:,0])*(bb2[:,1] - bb2[:,0]>0)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    denominator = (bb1_area + bb2_area - intersection_area)
    iou = intersection_area / (denominator*(denominator>0)+1e-5)
    iou = iou*(iou>0)
    iou = iou*(~torch.isnan(iou))
    return iou


# ---- align metrics - accuracy / f1 @ different IOU levels adapted from Sam's script --------

def calc_align_metrics(
        pred_boxes: np.ndarray, # (b, N)
        gt_boxes: np.ndarray,
        n_frames: int = None,
        inp_fmt: str = 'box',
        BACKGROUND_LABEL: int = -1,
        overlaps = (0.1, 0.25, 0.5),
):
    """Evaluate subtitle alignment quality.

    Args:
        pred_boxes: (b, 2) the predicted sub boxes in interval between [0,1] format
        gt_anno_paths: (b, 2) the gt sub boxes in interval between [0,1] format
        n_frames: the number of frames that correspond to the [0,1] interval - needed for box option only
        inp_fmt: the input format [box, vector]
    """

    correct = 0
    total = 0
    # total_subs = 0
    MAX_TIME_PAD_SECS = 10
    tp, fp, fn = np.zeros(len(overlaps)), np.zeros(len(overlaps)), np.zeros(len(overlaps))

    for pred_box, gt_box in zip(pred_boxes, gt_boxes):
        # pred_box = list(webvtt.read(pred_path))
        # gt_subs = list(webvtt.read(gt_box))
        # msg = (f"Expected num. preds {len(pred_box)} to match num. gt {len(gt_subs)} for"
        #        f" {pred_path}")
        assert len(pred_box) == len(gt_box)
        # total_subs += len(gt_subs)

        # To compute metrics, we convert all the subtitles into a sequence of frame-level
        # labels, where for each frame, the label indicates which subtitle (if any) it
        # was assigned.
        if inp_fmt == 'box':
            pred_frames = box2frames(
                box=pred_box,
                n_frames=n_frames,
                background_label=BACKGROUND_LABEL,
            )
            gt_frames = box2frames(
                box=gt_box,
                n_frames=n_frames,
                background_label=BACKGROUND_LABEL,
            )
        else:
            pred_frames = pred_box
            gt_frames = gt_box

        # Compute frame-level accuracy
        for pred, gt in zip(pred_frames, gt_frames):
            if gt == pred == BACKGROUND_LABEL: 
                continue
            total += 1
            if pred == gt:
                correct += 1

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
    # msg = (f"Frame-level accuracy: {100 * float(correct)/total:.2f}"
    #        f" (computed over {total} frames, {total_subs} subtitles)")

    # msg = (f"Frame-level accuracy: {100 * float(correct)/total:.2f}"
    #        f" (computed over {total} frames)")
    # print(msg)
    # for ii, overlap in enumerate(overlaps):
    #     precision = tp[ii] / float(tp[ii] + fp[ii])
    #     recall = tp[ii] / float(tp[ii] + fn[ii])
    #     f1 = 2.0 * (precision * recall) / (precision + recall)
    #     f1 = np.nan_to_num(f1) * 100
    #     print(f"F1@{overlap:0.2f}: {f1:.4f}")

    return correct, tp, fp, fn, total

    # import ipdb; ipdb.set_trace(context=20)

class F1Logger():

    def __init__(self, overlaps=(0.1, 0.25, 0.5), suffix=''):
        self.overlaps = overlaps
        n_ov = len(self.overlaps)
        self.acum ={f'tp{suffix}' : np.zeros(n_ov),
                    f'fp{suffix}': np.zeros(n_ov),
                    f'fn{suffix}': np.zeros(n_ov),
                    f'total_frames{suffix}': 0.0,
                    f'correct{suffix}': 0.0}  
        self.suffix = suffix

    def update(self, entry):
        for kk in self.acum:
            self.acum[kk] += entry[kk]

    @property
    def accuracy(self):
        return 100 * float(self.acum[f'correct{self.suffix}'])/self.acum[f'total_frames{self.suffix}']

    @property
    def f1(self):
        f1s = []
        tp, fp, fn = self.acum[f'tp{self.suffix}'], self.acum[f'fp{self.suffix}'], self.acum[f'fn{self.suffix}']
        for ii, overlap in enumerate(self.overlaps):
            precision = tp[ii] / float(tp[ii] + fp[ii])
            recall = tp[ii] / float(tp[ii] + fn[ii])
            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1) * 100
            f1s.append(f1)

        return f1s, self.overlaps


def box2frames(
        box: np.ndarray,
        n_frames: int,
        background_label: int,
) -> List[int]:
    """Convert subtitles into a single sequence of frames in which each subtitle is
    assigned a unique integer and the frames covered by that subtitle are assigned this
    integer.

    Args:
        subs: a list of webvtt caption objects, each of which represnts a subtitle caption
           with an associated start and end time as well as text.
        background_label: the value to be assigned to frames that are not covered by any
           subtitle (or frames covered by subtitles that are excluded).

    Returns:
        A list of frame-level sequence labels, each indicating the index of the subtitle
        which covered the current frame.
    """

    frames = [background_label for _ in range(n_frames)]

    # for sub_idx, caption in enumerate(subs):
    #     if sub_idx in exclude_subs:
    #         continue

    #     # Ensure that predicted caption alignment time falls within the bounds o fevaluation
    #     start_time = min(caption._start, max_time)
    #     end_time = min(caption._end, max_time)

    #     start_idx = round(fps * start_time)
    #     end_idx = round(fps * end_time)
    #     frames[start_idx:end_idx] = [sub_idx for _ in range(end_idx - start_idx)]

    # Ensure that predicted caption alignment time falls within the bounds o fevaluation
    start_time = max(min(box[0], 1),0)
    end_time = max(min(box[1], 1),0)

    start_idx = round(n_frames * start_time)
    end_idx = round(n_frames * end_time)
    sub_idx = 0
    frames[start_idx:end_idx] = [sub_idx for _ in range(end_idx - start_idx)]

    return frames

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
        try: 
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
        except: ## for when GT labels are all 0
            pass
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

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



def seconds_to_string(seconds):
    """
    For writing subtitles 
    """
    if seconds<0:
        print('WARNING seconds is less than 0')
        seconds = max(seconds,0)
    timestr = '0' + str(timedelta(seconds=seconds))
    if len(timestr) > 8:
        if len(timestr) != 8+7:
            print('ERROR timestr ', timestr)
            print('seconds', seconds)
        assert len(timestr) == 8 + 7, 'timestring '+timestr
        timestr = timestr[:-3]
    else:
        timestr = timestr + '.000'
    assert len(timestr) == 12, 'timestring '+timestr
    return timestr


def parse_vtt_sub_file(vtt_path):
    subs = webvtt.read(vtt_path)

    subs_parsed = []

    for sub in subs:

        x = datetime.strptime(sub.start, '%H:%M:%S.%f')
        time_fr = timedelta(hours=x.hour,
                            minutes=x.minute,
                            seconds=x.second,
                            microseconds=x.microsecond).total_seconds()

        x = datetime.strptime(sub.end, '%H:%M:%S.%f')
        time_to = timedelta(hours=x.hour,
                            minutes=x.minute,
                            seconds=x.second,
                            microseconds=x.microsecond).total_seconds()

        text = sub.text

        subs_parsed.append([text, time_fr, time_to])

    return subs_parsed

def get_spottings_episodes(data_paths, spottings_path): 

# reshape to get times, probs, names per episode 
    full = pickle.load(open(spottings_path, 'rb'))

    indices_of_episodes = [i for i in range(len(full['episode'])) if full['episode'][i].replace('--', '/') in data_paths]

    spottings_episodes = {}
    for i in indices_of_episodes: 
        if full['episode'][i].replace('--', '/') not in spottings_episodes.keys(): 
            spottings_episodes[full['episode'][i].replace('--', '/')] = {'words': [], 'times': []}
        word_list = full['annotations'][i]['words']
        time_list = full['annotations'][i]['times']
        spottings_episodes[full['episode'][i].replace('--', '/')]['words'].append(word_list)
        spottings_episodes[full['episode'][i].replace('--', '/')]['times'].append(time_list)

    for fname in data_paths: 
        spottings_episodes[fname]['words'] = [i for j in spottings_episodes[fname]['words'] for i in j]
        spottings_episodes[fname]['times'] = [i for j in spottings_episodes[fname]['times'] for i in j]

        # sort by time 
        spottings_episodes[fname]['words'] = [x for _,x in sorted(zip(spottings_episodes[fname]['times'],spottings_episodes[fname]['words']))]
        spottings_episodes[fname]['times'] = sorted(spottings_episodes[fname]['times'])

    return spottings_episodes

def sec2frame(time_sec: float, fps: int = 25):
    time_frame = int(fps * time_sec)
    return time_frame


def frame2index(time_frame: int, clip_size: int = 16, clip_stride: int = 8) -> int:
    """Assign each time to the clip with the nearest center"""
    assert clip_size == 16, "Do more sanity checks for other sizes"
    # handle boundary condition at start (TODO: handle boundary condition at the end)
    offset = clip_stride // 2
    if time_frame <= offset:
        time_index = 0
    else:
        time_frame -= offset
        time_index = int(time_frame // clip_stride)
    return time_index

def sec2index(
    time_sec: float, fps: int = 25, clip_size: int = 16, clip_stride: int = 8
) -> int:
    time_frame = sec2frame(time_sec=time_sec, fps=fps)
    time_index = frame2index(time_frame=time_frame, clip_size=clip_size, clip_stride=clip_stride)
    return time_index

def get_feature_interval(features, t0_sec, t1_sec, pad_sec=0, fps=25, clip_size=16, clip_stride=8):
    t0_ix = sec2index(time_sec=t0_sec - pad_sec, fps=fps, clip_size=clip_size, clip_stride=clip_stride)
    t1_ix = sec2index(time_sec=t1_sec + pad_sec, fps=fps, clip_size=clip_size, clip_stride=clip_stride)
    # TODO(include the t1_ix?)
    return t0_ix, t1_ix, features[t0_ix:t1_ix]

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
 
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    
    filtered_sentence = []
    
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    
    return ' '.join(filtered_sentence)


def calc_recall_word(hypotheses, references, pad_idx=2):
    """
    Recall per word instead of per sub
    Fraction of subtitle words which are correctly predicted 

    :param hypotheses: list of hypotheses (idx)
    :param references: list of references (idx)
    :return:
    """
    recall = 0
    total_words = 0
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        for word in ref:
            # this is the no-class index
            if word != pad_idx:
                total_words += 1 
                recall_word = len(set([word]).intersection(set(hyp)))
                assert recall_word == 0 or recall_word == 1 
                recall += recall_word
    if total_words == 0:
        return 0
    else:
        return recall / float(total_words)

def calc_recall_class(hypotheses, references, pad_idx=2):
    """
    Recall per class
    Fraction of subtitle words which are correctly predicted 

    :param hypotheses: list of hypotheses (idx)
    :param references: list of references (idx)
    :return:
    """
    classidx_2_recall_word = {}
    total_words = 0
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        for word in ref:
            # this is the no-class index
            if word != pad_idx:
                total_words +=1
                recall_word = len(set([word]).intersection(set(hyp)))
                assert recall_word == 0 or recall_word == 1 
                if word not in classidx_2_recall_word.keys():
                    classidx_2_recall_word[word] = []
                classidx_2_recall_word[word].append(recall_word)
    for word in classidx_2_recall_word.keys():
        classidx_2_recall_word[word] = np.average(classidx_2_recall_word[word])
    if total_words == 0:
        return 0
    else:
        return np.average(list(classidx_2_recall_word.values()))

def wer(hypotheses, references, pad_idx=2, CTC=False):
    """
    word error rate 
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    # filter those which are empty because we've removed stop words
    idx_keep = np.where(np.asarray(references)!=pad_idx)
    # filter those which are empty because we've removed stop words
    filter_references = list(np.asarray(references)[idx_keep])
    if not CTC: 
        hypotheses = list(np.asarray(hypotheses)[idx_keep])
        error = wer_calc(" ".join([str(int(i)) for i in filter_references]), " ".join([str(i) for i in hypotheses]))
    else: 
        error = wer_calc(" ".join([str(i) for i in filter_references]), " ".join([str(i) for i in np.concatenate(hypotheses)]))    
    return error

def shift_spottings(ann_types, ann_times, fps):
    new_times = []
    #  M* [-9, 11], D* [-3, 22], P [0, 19], E [0, 19], N [0, 19]. 
    for idx, ann_t in enumerate(ann_types):
        if ann_t == "M*":
            shift = 1/float(fps)
            new_times.append(ann_times[idx]+shift)
        elif ann_t ==  "D*":
            shift = 9.5/float(fps)
            new_times.append(ann_times[idx]+shift)
        elif ann_t == "A":
            shift = 0/float(fps)
            new_times.append(ann_times[idx]+shift)
        elif ann_t == "P":
            shift = 9.5/float(fps)
            new_times.append(ann_times[idx]+shift)
        elif ann_t == "E":
            shift = 9.5/float(fps)
            new_times.append(ann_times[idx]+shift)
        elif ann_t == "N":
            shift = 9.5/float(fps)
            new_times.append(ann_times[idx]+shift)
        else: 
            new_times.append(ann_times[idx])
            print(f'WARNING invalid type {ann_t}, no shift applied')
    return new_times