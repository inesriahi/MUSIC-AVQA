import ast
import json
import os
import random
import warnings

import numpy as np
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from parallelzipfile import ParallelZipFile

warnings.filterwarnings('ignore')


def ids_to_multinomial(id, categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]

class AVQA_dataset(Dataset):
    """
    Dataset class for Audio-Visual Question-Answering (AVQA).

    Attributes:
        ques_vocab (list): Vocabulary for questions.
        ans_vocab (list): Vocabulary for answers.
        word_to_ix (dict): Word to index mapping.
        samples (list): List of samples loaded from JSON.
        max_len (int): Maximum length of a question.
        audio_dir (str): Directory for audio data.
        video_res14x14_dir (str): Directory for video data in resolution 14x14.
        transform (callable): Optional transformation to apply on video frames.
        video_list (list): List of unique video names.
        video_len (int): Total number of video frames.
        my_normalize (callable): Transformation to normalize images.
        norm_mean (float): Mean for audio normalization.
        norm_std (float): Standard deviation for audio normalization.
    """
    def __init__(self, 
                 label, 
                 audio_dir, 
                 video_res14x14_dir, 
                 transform=None, 
                 mode_flag='train'):
        """
        Initialize the dataset.

        Args:
            label (str): Path to the JSON file containing samples.
            audio_dir (str): Directory for audio data.
            video_res14x14_dir (str): Directory for video data in resolution 14x14.
            transform (callable, optional): Optional transformation to apply on self.samples.
            mode_flag (str, optional): Mode ('train' or 'test'). Default is 'train'.
        """
        
        samples = json.load(open('/scratch/project_462000189/datasets/MUCIS-AVQA/json/avqa-train.json', 'r'))
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14    # question length

        self.audio_dir = audio_dir
        self.video_res14x14_dir = video_res14x14_dir
        self.transform = transform

        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list
        self.video_len = 60 * len(video_list)
        
        self.my_normalize = Compose([
            Resize([224,224], interpolation=Image.BICUBIC),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        
        ### ---> for AVQA
        self.norm_mean =  -5.187877655029297
        self.norm_std =  3.8782312870025635
        
    # audio
    def wavform2fbank(self, filename, num_secs=60):
        """
        Convert a waveform to filterbank (fbank) features.

        Args:
            filename (str): Path to the audio file.
            num_secs (int, optional): Duration in seconds to consider. Default is 60.

        Returns:
            torch.Tensor: Tensor of shape (T, len, dim) where T=10, len=224, dim=224.
        """
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=224, dither=0.0, frame_shift=4.45)
        
        target_length = 224 # to match ViT
        
        ## align
        if len(fbank) > num_secs*target_length:
            fbank = fbank[:num_secs*target_length]
            
        sample_indices = np.linspace(0, len(fbank) - 6*target_length, num=10, dtype=int)
        total_audio = []
        for audio_idx in sample_indices:
            audio_sample = fbank[audio_idx:audio_idx+int(target_length)]
            ########### ------> very important: audio normalized
            audio_sample = (audio_sample - self.norm_mean) / (self.norm_std * 2)
            ### <--------
            n_frames = audio_sample.shape[0]
            p = target_length - n_frames

            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p)) # padding bottom by p
                audio_sample = m(audio_sample)
            elif p < 0:
                audio_sample = audio_sample[:target_length, :]

            total_audio.append(audio_sample)

        return torch.stack(total_audio) # (T, len, dim) T = 10, len = total_length = 224, dim = num_mel_bins = 224

    # visual
    def get_frames(self, zip_file):
        """
        Extract frames from a zip file containing images.

        Args:
            zip_file (ZipFile): Zip file containing images.

        Returns:
            torch.Tensor: Tensor of shape (T, C, H, W) where T=10, C=3, H=W=224.
        """
        total_num_frames = len(zip_file.namelist())
        
        sample_indices = np.linspace(1, total_num_frames, 10, dtype=int)
        total_img = []
        for index in sample_indices:
            # Format the filename with leading zeros (e.g., "000001.jpg")
            filename = f"{index:06d}.jpg"
            
            # Read the file from the zip archive
            buffer = zip_file.read(filename)
            
            # Decode the image and normalize its values to [0, 1]
            image = torchvision.io.decode_image(torch.frombuffer(buffer, dtype=torch.uint8)) / 255.0
            
            normalized_image = self.my_normalize(image)
            
            total_img.append(normalized_image)
        return torch.stack(total_img) # (T, C, H, W) T = 10, C = 3, H = W = 224
    
    def get_negative_sample(self, current_video_index):
        """
        Get a negative sample (image) from a different video than the current one.

        Args:
            current_video_index (int): Index of the current video, to ensure negative sample comes from a different video.

        Returns:
            torch.Tensor: The negative sample image tensor.
        """
        while True:
            # Randomly select a frame ID
            neg_frame_id = random.randint(0, self.video_len - 1)
            # Check if the randomly selected frame belongs to a different video
            if int(neg_frame_id / 60) != current_video_index:
                break
            
        # Get the video and frame indices for the negative sample
        neg_video_id = int(neg_frame_id / 60)
        neg_frame_flag = neg_frame_id % 60
        
        neg_video_name = self.video_list[neg_video_id]
        
        zip_file_n = ParallelZipFile(self.video_res14x14_dir+'/'+neg_video_name+'.zip', 'r')
        
        # Get the total number of frames in the negative video
        total_num_frames_n = len(zip_file_n.namelist())
        
        # Calculate the sample index
        sample_indx_n = np.linspace(1, total_num_frames_n, num=60, dtype=int)
        buffer = zip_file_n.read(str("{:06d}".format(sample_indx_n[neg_frame_flag])) + '.jpg')
        
        # Decode and normalize the image
        tmp_img_n = torchvision.io.decode_image(torch.frombuffer(buffer, dtype=torch.uint8)) / 255.0
        visual_nega_clip = self.my_normalize(tmp_img_n)

        return visual_nega_clip

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]
        name = sample['video_id']
        # audio = np.load(os.path.join(self.audio_dir, name + '.npy'))#60*512
        # audio = audio[::6, :]

        total_audio = self.wavform2fbank(self.audio_dir+'/'+name+ '.wav', num_secs=60)

        # visual_out_res18_path = '/home/guangyao_li/dataset/avqa-features/visual_14x14'
        # visual_posi = np.load(os.path.join(self.video_res14x14_dir, name + '.npy'))

        # visual_posi [60, 512, 14, 14], select 10 frames from one video
        # visual_posi = visual_posi[::6, :]

        ### ---> video frame process
        ###
        zip_file = ParallelZipFile(self.video_res14x14_dir+'/'+name+'.zip', 'r')
        total_img = self.get_frames(zip_file)

        video_idx = self.video_list.index(name)
        visual_nega = [self.get_negative_sample(video_idx) for _ in range(total_img.shape[0])]
        visual_nega = torch.stack(visual_nega)
        # visual nega [60, 512, 14, 14]

        # question
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # answer
        answer = sample['anser']
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        sample = {'audio': total_audio, 'visual_posi': total_img, 'visual_nega': visual_nega, 'question': ques, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):

    def __call__(self, sample):

        audio = sample['audio']
        visual_posi = sample['visual_posi']
        visual_nega = sample['visual_nega']
        label = sample['label']

        return { 
            'audio': sample['audio'],
            'visual_posi': sample['visual_posi'],
            'visual_nega': sample['visual_nega'],
            'question': sample['question'],
            'label': label
        }