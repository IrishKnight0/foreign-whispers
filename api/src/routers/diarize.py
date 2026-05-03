# extracts audio from a video, and runs pyannote diarization on it
# also merges speaker labels into transcription

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService



router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diardir = settings.diarizations_dir
    diardir.mkdir(parents=True, exist_ok=True)
    diarpath = diardir / f"{title}.json"



    if diarpath.exists():
        data = json.loads(diarpath.read_text())
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    video_path = settings.videos_dir / f"{title}.mp4"
    audio_path = diardir / f"{title}.wav"
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", str(audio_path)],
        check=True
    )


    diarseg = _alignment_service.diarize(str(audio_path))

    speakers = sorted(set(s["speaker"] for s in diarseg))

    result = {"speakers": speakers, "segments": diarseg}
    diarpath.write_text(json.dumps(result))




    from foreign_whispers.diarization import assign_speakers


    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
        labeled_segments = assign_speakers(transcript.get("segments", []), diarseg)
        transcript["segments"] = labeled_segments
        transcript_path.write_text(json.dumps(transcript))

        return DiarizeResponse(video_id=video_id, speakers=speakers, segments=diarseg)