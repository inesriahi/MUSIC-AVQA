# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy
from scipy.io import wavfile

import mel_features
import vggish_params


def waveform_to_examples(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1) # If data has more than one channel (is stereo), it is converted to mono by averaging across channels. The shape changes from (num_samples, num_channels) to (num_samples,).
    
  # Resample to the rate assumed by VGGish (16000).
  if sample_rate != vggish_params.SAMPLE_RATE: # If the sample rate of data does not match vggish_params.SAMPLE_RATE, data is resampled to this rate. The number of samples in data may change as a result.
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ) # The log mel spectrogram of data is computed, resulting in a 2-D numpy array log_mel of shape (num_frames, num_mel_bins). The exact shape depends on the parameters defined in vggish_params and the length of data.

  # Frame features into examples.
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length) # log_mel is framed into examples of a specified window length and hop length, resulting in a 3-D numpy array log_mel_examples of shape (num_examples, example_window_length, num_mel_bins).
  return log_mel_examples


def wavfile_to_examples(wav_file, num_secs):
  """Convenience wrapper around waveform_to_examples() for a common WAV format.

  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.

  Returns:
    See waveform_to_examples.
  """
  sr, snd = wavfile.read(wav_file) # sr (sampling rate) is an integer representing the number of samples per second in the audio data. snd is a numpy array containing the audio data. Its shape depends on the audio: 
  # For mono audio: (num_samples,)
  # For stereo audio: (num_samples, num_channels)
  # L = sr * 60 
  L = sr * num_secs # L is the total number of samples to process, calculated as sampling rate * number of seconds.
  ch = 1
  if len(snd.shape) >1:
      ch = snd.shape[1]
  wav_data = np.zeros((L, ch))

  # if snd.shape[0] < sr * 60:
  #     wav_data[:snd.shape[0], :] = snd
  # else:
  #     wav_data = snd[:L, :]

  wav_data = snd[:L, :] # The relevant section of snd is sliced to have L samples and assigned to wav_data. The shape of wav_data is (L, ch).

  wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0] Normalization
  # T = 60
  T = num_secs
  L = wav_data.shape[0]
  log_mel = np.zeros([T, 96, 64])

  # print("\nT: ", T)
  # print("L: ", L)
  # print("log_mel: ", log_mel.shape)

  for i in range(T): # Loop for Computing log_mel Values: T times (each iteration corresponds to one second of audio data).
      s = i * sr # start of segment
      e = (i + 1) * sr # end of segment
      if len(wav_data.shape) > 1:
          data = wav_data[s:e, :]
      else:
          data = wav_data[s:e]

      # print("\ns-e: ", s, e)
      # print("data input: ", data.shape)
      log_mel[i, :, :] = waveform_to_examples(data, sr) # The result is assigned to the corresponding slice of log_mel.

  # print("log mel: ", log_mel.shape)
  return log_mel # shape (T, 96, 64) is returned. Where T is the number of seconds
