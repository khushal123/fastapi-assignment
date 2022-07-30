import os
from typing import Any, List, Iterator, Tuple

import torch
import torchaudio
import torchaudio.transforms as transforms
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging


app = FastAPI()
logger = logging.getLogger("app")


class Model:
    """A mocked model class that gives random predictions"""

    def __call__(self, audio: torch.tensor) -> float:
        """A random prediction method that predicts float values between 0 and 1

        Args:
            audio: Tensor representation of the audio file

        Returns:
            Float value representing the confidence of the prediction
        """
        return float(torch.rand(1))


class Prediction(BaseModel):
    phrase: str
    time: int
    confidence: float


MODEL_CONFIDENCE_THRESHOLD = 0.9
SAMPLE_RATE = 8000
MODEL_DICT = {
    "call": Model(),
    "is": Model(),
    "recorded": Model(),
}


@app.get("/")
def root():
    logger.warn("root")
    return {"message": "Hello world"}


@app.get("/api/detect/", response_model=List[Prediction])
def generate_phrase_detections(utterance: str, audio_loc: str) -> Any:
    """Run inference on an audio file with a model for an utterance. Currently
    available utterances are: "call", "is", "recorded"

    Args:
        utterance: Case sensitive name of the model to be used for inference
        audio_loc: The full or relative path to the audio file for which inference
            is to be executed
    """
    try:
        model = MODEL_DICT[utterance]
    except KeyError:
        raise HTTPException(
            404, f"Utterance {utterance} not found in local model dictionary"
        )

    try:
        audio_loc = os.path.join(os.getcwd(), "audio", audio_loc)
        resampled_audio = load_resampled(audio_loc, SAMPLE_RATE)
    except FileNotFoundError:
        raise HTTPException(404, f"File {audio_loc} not found")
    predictions = []
    for time, audio_snip in iterate_call(resampled_audio):
        confidence = model(audio_snip)
        if confidence > MODEL_CONFIDENCE_THRESHOLD:
            predictions.append(
                Prediction(
                    phrase=utterance, time=time / SAMPLE_RATE, confidence=confidence
                )
            )

    
    return predictions


def load_resampled(audio_loc: str, resample_rate: int = 8000) -> torch.tensor:
    """Load and resample an audio file

    Args:
        audio_loc: Full or relative path to the audio file
        resample_rate: What sampling rate should the audio file be resampled
            to. Defaults to 8000

    Returns:
        torch.tensor with loaded audio file data

    Raises:
        FileNotFoundError: If the audio_loc is not a valid audio file
    """
    try:
        print(audio_loc)
        audio, rate = torchaudio.load(audio_loc)
    except RuntimeError as e:
        raise FileNotFoundError(e)

    resampler = transforms.Resample(rate, resample_rate)
    return resampler(audio)


def iterate_call(
    audio: torch.tensor, stride: int = 8000, window: int = 8000
) -> Iterator[Tuple[int, torch.tensor]]:
    """Iterate over an audio tensor in with given stride and window

    Args:
        audio: The tensor containing audio file data
        stride: The amount of samples to move at each iteration
        window: The amount of samples to include in the cut audio

    Yields:
        A tuple containing the starting time index of the audio snipet and
            a tensor containing the audio data
    """
    for start in range(audio.shape[-1] // stride):
        start_idx = start * stride
        yield start_idx, audio[:, start_idx: start_idx + window]
