# this is made to find the right reference voice wav file for a given speaker and language

from pathlib import Path


def resolve_speaker_wav(
    speakers_dir: Path,
    lang: str,
    speaker_id: str | None = None,
) -> str | None:


    if speaker_id:
        path_speaker = speakers_dir / lang / f"{speaker_id}.wav"
        
        if path_speaker.exists():
            return f"{lang}/{speaker_id}.wav"


    lang_default = speakers_dir / lang / "default.wav"
    
    if lang_default.exists():
        return f"{lang}/default.wav"


    global_default = speakers_dir / "default.wav"
    
    if global_default.exists():
        return "default.wav"



    return None