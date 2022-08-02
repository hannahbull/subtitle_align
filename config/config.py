import configargparse

def save_opts(args, fn):
    with open(fn, 'w') as fw:
        for items in vars(args):
            fw.write('%s %s\n' % (items, vars(args)[items]))

def str2bool(v):
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    elif v in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean argument needs to be true or false. '
                        'Instead, it is %s.' % v)

def load_opts():

    parser = configargparse.ArgumentParser(description="main")
    parser.register('type', bool, str2bool)

    parser.add('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--gpu_id', type=str, default=0, help='-1: all, 0-7: GPU index')

    parser.add_argument('--resume', type=str, default=None, help='', nargs='+')
    parser.add_argument('--test_only', action='store_true', help='Run only evaluation')
    parser.add_argument('--n_workers', type=int, default=0, help='Num data workers')

    parser.add_argument('--model', type=str, default='gt_align_invtransformer', help='Model type')
    parser.add_argument('--dataset', type=str, default='video_text', help='Dataset type')
    parser.add_argument('--trainer', type=str, default='video_text', help='Trainer type')

    ### --- dataloader 
    parser.add_argument('--train_videos_txt', type=str, default='data/bobsl_align_train.txt', help='txt file with one line per video name for training set')
    parser.add_argument('--val_videos_txt', type=str, default='data/bobsl_align_val.txt', help='txt file with one line per video name for validation set')
    parser.add_argument("--test_videos_txt", type=str, default='data/bobsl_align_test.txt', help='txt file with one line per video name for test set')

    parser.add_argument('--random_subset_data', type = int, default = 1e9, help = "If N < number of videos, random subset of N videos from training data")
    parser.add_argument('--random_subset_data_seed', type = int, default = 123, help = "Seed of random subset of training data")
    
    # load subtitles or load words
    parser.add_argument('--load_subtitles', type = bool, default=True, help='Load subtitles texts and times (e.g. for training on subtitles)')
    parser.add_argument('--load_words', type = bool, default=False, help='Load word spottings texts and times (e.g. for word pre-training)')
    parser.add_argument('--load_features',  type = bool, default=True, help='Load features')

    parser.add_argument('--features_path', 
                            type = str, 
                            default = '/scratch/shared/beegfs/gul/datasets/features/bobsl/featurize-c2281_16f_pad10sec_m8_-15_4_d0.8_-3_22_anon-v0-stride0.25/filtered/', 
                            help = 'Path to I3D features directory')
    parser.add_argument('--gt_sub_path', 
                            type = str, 
                            default = '/scratch/shared/beegfs/albanie/shared-datasets/bobsl/public_dataset_release/subtitles/manually-aligned/', 
                            help = 'Path to aligned subtitles directory')
    parser.add_argument('--pr_sub_path',
                            type = str, 
                            default = '/scratch/shared/beegfs/hbull/shared-datasets/bobsl/audio-aligned-corrected/', 
                            help = 'Path to original un-aligned subtitles directory. Empty string for no prior subtitles.')
    parser.add_argument('--spottings_path', 
                            type=str, 
                            default='/scratch/shared/beegfs/hbull/data/annotationsMDAPEN.pkl')

    parser.add_argument("--pool_feats", type = bool, default = False, help='Average features')
    parser.add_argument('--input_features_stride', type=int, default=4, help='Feature file contains this stride value')
    parser.add_argument('--subsample_stride', type=int, default=1, help='Sample every Nth frame from input features')
    parser.add_argument('--pad_start_features', action='store_true', help='Pad START rather than END of features if necessary')
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument('--shuffle_getitem', type = bool, default = False, help = "Shuffle get item")

    parser.add_argument('--pr_subs_delta_bias', type=float, default=0, help='Bias to add to prior subtitles in seconds')
    parser.add_argument('--gt_subs_delta_bias', type=float, default=0, help='Bias to add to GT subtitles in seconds')
    parser.add_argument('--words_delta_bias', type=float, default=0, help='Bias to add to words in seconds')
    parser.add_argument('--shift_spottings', type=bool, default=True, help='Shift spottings to pre-defined location')

    # word spotting parameters
    parser.add_argument('--pad_annot', type=float, default=0.5, help='Padding in seconds (left and right) of word spotting annotation times')
    parser.add_argument('--conf_thresh_annot', type=float, default=0.8, help='Confidence threshold for word spottings (excluding A, E, N)')

    # percentage of negative windows to add
    parser.add_argument('--negatives_percent', type = float, default = 0, help = "Proportion of negatives (GT=0) to add")
    parser.add_argument('--centre_window', action='store_true', help='Centre search window around prior')

    parser.add_argument('--jitter_abs', action='store_true', help='Whether to jitter absolutely or relative to length of sub/spotting')
    parser.add_argument('--jitter_location', action='store_true', help='Whether to jitter forward or backward in time')
    parser.add_argument('--jitter_loc_quantity', type=float, default=0, help='Percentage of subtitle width to jitter by OR if jitter_abs maximum shift in seconds')
    parser.add_argument('--jitter_width_secs', type=int, default=0, help='jitter width of prior sub in s')

    parser.add_argument('--fixed_feat_len', type = float, default = 20, help = "Feature length in seconds")

    parser.add_argument('--max_text_len_filter', type=int, default=1e9, help='Maximum text length (FILTER)')
    parser.add_argument('--max_sent_len_filter', type = float, default = 1e9, help = 'Max length of vid in seconds (FILTER) ')    
    parser.add_argument('--min_text_len_filter', type=int, default=0, help='Min text length (FILTER)')
    parser.add_argument('--min_sent_len_filter', type = float, default = 0, help = 'Min length of length of vid in seconds (FILTER) ')    
    parser.add_argument('--max_feat_len', type = int, default = 0, help = 'Max length of features (CROPPING) ')    
    parser.add_argument('--max_text_len', type=int, default=0, help='Maximum text length (CROPPING)')

    # text cleaning and augmentation
    parser.add_argument('--stem_words', type=bool, default=False, help='Stem words in the subtitle')
    parser.add_argument('--lemmatize_words', type=bool, default=False, help='Lemmatize words in the subtitle')
    parser.add_argument('--remove_stopwords', type = bool, default=True, help='Remove stopwords before further text processing')
    parser.add_argument('--shuffle_words_subs', type = float, default = 0, help = "Percentage of subtitles to shuffle words in subtitles during training")
    parser.add_argument('--drop_words_subs', type = float, default = 0, help = "Percentage to drop words in subtitles")

    # feature augmentation
    parser.add_argument('--drop_feats', type = float, default = 0, help = "Percentage to drop visual features")
    parser.add_argument('--shuffle_feats', type = float, default = 0, help = "Percentage to shuffle visual features")

    # --- model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Transformer model dimension')
    parser.add_argument('--n_enc_layers', type=int, default=2, help='Transformer number encoder layers')
    parser.add_argument('--n_dec_layers', type=int, default=2, help='Transformer number decoder layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Transformer dim')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Transformer dropout')
    parser.add_argument('--positional_encoding_text', type=bool, default=False, help='Add positional encodings to text')

    parser.add_argument('--concatenate_prior', type=bool, default=True, help='Concatenate prior location')
    parser.add_argument('--finetune_bert', type=bool, default=False, help='Finetune Bert')

    # --- trainer
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or adamw optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--pos_weight', type=float, default=0, help='Add pos weights to CrossEntropyLoss')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='Grad clip norm')

    # --- save output
    parser.add_argument('--save_every_n', type=int, default=1, help='validate and save checkpoint every nth epoch')

    parser.add_argument('--save_path', type=str, default="inference_output/", help='Folder to save model arguments')

    parser.add_argument("--save_vtt", type = bool, default = False, help="Save subtitles as vtt")
    parser.add_argument('--save_probs', type = bool, default = False, help = "Save CE softmax outputs ")
    parser.add_argument("--dtw_postpro", type = bool, default = False, help='Run DTW postprocessing')

    parser.add_argument('--save_probs_folder', type=str, default='inference_output/probabilities', help='Folder to save probabilities')
    parser.add_argument('--save_subs_folder', type=str, default='inference_output/subtitles', help='Folder to save subtitles')
    parser.add_argument('--save_postpro_subs_folder', type=str, default='inference_output/subtitles_postprocessing', help='Folder to save subtitles with postprocessing to remove overlaps')


    args = parser.parse_args()

    return args