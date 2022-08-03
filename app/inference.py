import os
from typing import Any, List, Iterator, Tuple
from pydantic import BaseModel

import torch
import torchaudio
import torchaudio.transforms as transforms
from fastapi import FastAPI, HTTPException
from app.config import settings
from app.database.base import Session
from fastapi.params import Depends
from app.database.confidence.models import Prediction
from app.database.confidence.service import AppService
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


class PredictionModel(BaseModel):
    phrase: str
    time: int
    confidence: float


MODEL_DICT = {
    "call": Model(),
    "is": Model(),
    "recorded": Model(),
}


def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    logger.warn("root")
    return {"message": "Hello world"}


@app.get("/api/detect/", response_model=List[PredictionModel])
def generate_phrase_detections(utterance: str, audio_path: str, db: Session = Depends(get_db)) -> Any:
    """Run inference on an audio file with a model for an utterance. Currently
    available utterances are: "call", "is", "recorded"

    Args:
        utterance: Case sensitive name of the model to be used for inference
        audio_path: The full or relative path to the audio file for which inference
            is to be executed
    """
    try:
        model = MODEL_DICT[utterance]
    except KeyError:
        raise HTTPException(
            404, f"Utterance {utterance} not found in local model dictionary"
        )

    try:
        audio_loc = os.path.join(f"{os.getcwd()}/app", audio_path)
        resampled_audio, duration = load_resampled(
            audio_loc, settings.SAMPLE_RATE)
    except FileNotFoundError:
        raise HTTPException(404, f"File {audio_loc} not found")

    media_file = None
    service = AppService(db)
    try:
        media_file = service.get_media_by_name(audio_path, duration)
    except Exception as e:
        return HTTPException(500, "Internal server error")

    predictions = []
    prediction_response = []
    for time, audio_snip in iterate_call(resampled_audio):
        confidence = model(audio_snip)
        if confidence > settings.MODEL_CONFIDENCE_THRESHOLD:
            predictions.append(
                Prediction(
                    phrase=utterance, time=time / settings.SAMPLE_RATE, confidence=confidence,
                    media_id=media_file.id
                )
            )
            prediction_response.append(
                PredictionModel(phrase=utterance, time=time /
                                settings.SAMPLE_RATE, confidence=confidence)
            )
    save_predictions = service.save_bulk(predictions=predictions)
    return prediction_response


@app.get("/api/confidence/{page}/{limit}")
def get_audio_confidences(page: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """Return list of audio files and their confidence scores. 
    """
    try:
        service = AppService(db)
        return service.get_confidence_list(page=page, limit=limit)
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            404, "Something went wrong"
        )


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
        audio, rate = torchaudio.load(audio_loc)
        num_frames = torchaudio.info(audio_loc).num_frames
        duration = num_frames/rate
    except RuntimeError as e:
        raise FileNotFoundError(e)

    resampler = transforms.Resample(rate, resample_rate)
    return [resampler(audio), duration]


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
