python main.py \
--features_path '/scratch/shared/beegfs/gul/datasets/features/bobsl/featurize-c2281_16f_pad10sec_m8_-15_4_d0.8_-3_22_anon-v0-stride0.25/filtered/' \
--gt_sub_path '/scratch/shared/beegfs/albanie/shared-datasets/bobsl/public_dataset_release/subtitles/manually-aligned/' \
--pr_sub_path '/scratch/shared/beegfs/hbull/shared-datasets/bobsl/audio-aligned-corrected/' \
--gpu_id 0 \
--n_workers 32 \
--batch_size 1 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro True \
--resume 'inference_output/finetune_subtitles/checkpoints/model_0000264041.pt' \

# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 70.89 F1@0.10: 74.08 F1@0.25: 66.78 F1@0.50: 53.22

### 2.7s shift baseline
# frame_accuracy: 62.40
# f1_10: 72.77
# f1_25: 64.08
# f1_50: 44.60

