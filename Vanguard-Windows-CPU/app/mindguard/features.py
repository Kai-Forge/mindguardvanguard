"""Dual-channel Log-Mel extraction and SpecAugment."""

import random

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T

from .config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE, WIN_LENGTH


def make_mel_transform() -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        f_min=0,
        f_max=SAMPLE_RATE // 2,
        power=2.0,
    )


def waveforms_to_logmel_batch(
    waveforms,
    mel_transform,
    top_db: float = 80.0,
    frame_lengths: torch.Tensor | None = None,
    power_db_multiplier: float = 20.0,
):
    """Convert padded waveforms to Log-Mel features.

    ``power_db_multiplier=20`` preserves the historical baseline even though
    the Mel transform returns power.  New experiments can select the standard
    power definition (10).  When ``frame_lengths`` is provided, feature frames
    created only by waveform batch padding are reset to zero so the CNN sees
    the same right boundary as it sees for a sample evaluated alone.
    """
    batch, channels, samples = waveforms.shape
    mel = mel_transform(waveforms.reshape(batch * channels, samples))
    mel = torch.clamp(mel, min=1e-10)
    log_mel = power_db_multiplier * torch.log10(mel)
    maximum = log_mel.reshape(batch * channels, -1).max(dim=1).values.view(-1, 1, 1)
    log_mel = torch.clamp(log_mel, min=maximum - top_db)
    log_mel = log_mel.reshape(batch, channels, log_mel.shape[1], log_mel.shape[2])
    if frame_lengths is not None:
        steps = torch.arange(log_mel.shape[-1], device=log_mel.device).view(1, 1, 1, -1)
        valid = steps < frame_lengths.to(log_mel.device).view(-1, 1, 1, 1)
        log_mel = log_mel.masked_fill(~valid, 0.0)
    return log_mel


def select_modality(specs: torch.Tensor, modality: str) -> torch.Tensor:
    """Select speech, EGG, or both while preserving a channel dimension."""
    if modality == "both":
        return specs
    if modality == "speech":
        return specs[:, 0:1]
    if modality == "egg":
        return specs[:, 1:2]
    raise ValueError(f"Unsupported modality: {modality}")


def waveforms_to_logmel_variable(
    waveforms: torch.Tensor,
    sample_lengths: torch.Tensor,
    mel_transform,
    top_db: float = 80.0,
    power_db_multiplier: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract each utterance before padding, then pad features with CNN zeros.

    Computing the STFT after waveform padding changes the right-edge reflection
    context.  That made predictions depend on what other utterances happened to
    share the batch.  This function removes that dependency and returns the
    exact feature-frame lengths produced by the transform.
    """
    utterance_features = []
    frame_lengths = []
    for waveform, sample_length in zip(waveforms, sample_lengths.tolist()):
        trimmed = waveform[:, :sample_length]
        mel = mel_transform(trimmed)
        mel = torch.clamp(mel, min=1e-10)
        log_mel = power_db_multiplier * torch.log10(mel)
        maximum = log_mel.reshape(log_mel.shape[0], -1).max(dim=1).values.view(-1, 1, 1)
        log_mel = torch.clamp(log_mel, min=maximum - top_db)
        utterance_features.append(log_mel.permute(2, 0, 1))
        frame_lengths.append(log_mel.shape[-1])
    padded = pad_sequence(utterance_features, batch_first=True, padding_value=0.0)
    return padded.permute(0, 2, 3, 1), torch.tensor(frame_lengths, dtype=torch.long)


def spec_augment(spec, freq_mask: int, time_mask: int, n_freq: int, n_time: int):
    output = spec.clone()
    for index in range(output.shape[0]):
        for _ in range(n_freq):
            width = random.randint(1, freq_mask)
            start = random.randint(0, output.shape[2] - width)
            region = output[index, :, start:start + width, :]
            output[index, :, start:start + width, :] = region.mean()
        for _ in range(n_time):
            width = random.randint(1, time_mask)
            start = random.randint(0, output.shape[3] - width)
            region = output[index, :, :, start:start + width]
            output[index, :, :, start:start + width] = region.mean()
    return output
