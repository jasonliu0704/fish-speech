import base64
import wave

import ormsgpack
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play

from fish_speech.utils.file import audio_to_bytes, read_ref_text
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


def process_references(reference_id=None, reference_audio=None, reference_text=None):
    if reference_id is None:
        byte_audios = []
        ref_texts = []
        if reference_audio:
            byte_audios = [audio_to_bytes(ref_audio) for ref_audio in reference_audio]
        if reference_text:
            ref_texts = [read_ref_text(ref_text) for ref_text in reference_text]
        return byte_audios, ref_texts
    return [], []

def create_tts_request(text, reference_id=None, reference_audio=None, reference_text=None, **kwargs):
    byte_audios, ref_texts = process_references(reference_id, reference_audio, reference_text)
    
    data = {
        "text": text,
        "references": [
            ServeReferenceAudio(
                audio=ref_audio if ref_audio is not None else b"", text=ref_text
            )
            for ref_text, ref_audio in zip(ref_texts, byte_audios)
        ],
        "reference_id": reference_id,
        **kwargs
    }
    
    return ServeTTSRequest(**data)

def handle_streaming_response(response, output_file, channels=1, rate=44100):
    p = pyaudio.PyAudio()
    audio_format = pyaudio.paInt16
    stream = p.open(
        format=audio_format, channels=channels, rate=rate, output=True
    )

    wf = wave.open(f"{output_file}.wav", "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(rate)

    try:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                stream.write(chunk)
                wf.writeframesraw(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()

def synthesize_speech(
    text,
    url="http://34.170.108.64:8080/v1/tts",
    output="generated_audio",
    play_audio=False,
    streaming=False,
    api_key="YOUR_API_KEY",
    **kwargs
):
    request_data = create_tts_request(text=text, **kwargs)
    
    response = requests.post(
        url,
        data=ormsgpack.packb(request_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        stream=streaming,
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/msgpack",
        },
    )

    if response.status_code == 200:
        if streaming:
            handle_streaming_response(response, output, 
                                   channels=kwargs.get('channels', 1),
                                   rate=kwargs.get('rate', 44100))
        else:
            audio_format = kwargs.get('format', 'wav')
            audio_path = f"{output}.{audio_format}"
            with open(audio_path, "wb") as audio_file:
                audio_file.write(response.content)

            if play_audio:
                audio = AudioSegment.from_file(audio_path, format=audio_format)
                play(audio)
            return audio_path
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())
        return None

# Example usage:
if __name__ == "__main__":
    # Basic usage
    synthesize_speech(
        text="Hello, this is a test.",
        normalize=True,
        format="wav",
        max_new_tokens=1024,
        temperature=0.7
    )

    # With reference audio
    synthesize_speech(
        text="Hello, this is a test with reference.",
        # reference_audio=["path/to/reference.wav"],
        # reference_text=["Reference text"],
        streaming=True
    )
