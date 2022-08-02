from .video_text_dataset import VideoTextDataset

dataset_dict = {
    'video_text': VideoTextDataset,
}

__all__ = ['dataset_dict']