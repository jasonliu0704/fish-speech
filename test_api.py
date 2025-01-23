if __name__ == "__main__":

    args = parse_args()

    idstr: str | None = args.reference_id
    # priority: ref_id > [{text, audio},...]
    if idstr is None:
        ref_audios = args.reference_audio
        ref_texts = args.reference_text
        if ref_audios is None:
            byte_audios = []
        else:
            byte_audios = [audio_to_bytes(ref_audio) for ref_audio in ref_audios]
        if ref_texts is None:
            ref_texts = []
        else:
            ref_texts = [read_ref_text(ref_text) for ref_text in ref_texts]
    else:
        byte_audios = []
        ref_texts = []
        pass  # in api.py

    data = {
        "text": args.text,
        "references": [
            ServeReferenceAudio(
                audio=ref_audio if ref_audio is not None else b"", text=ref_text
            )
            for ref_text, ref_audio in zip(ref_texts, byte_audios)
        ],
        "reference_id": idstr,
        "normalize": args.normalize,
        "format": args.format,
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "streaming": args.streaming,
        "use_memory_cache": args.use_memory_cache,
        "seed": args.seed,
    }

    pydantic_data = ServeTTSRequest(**data)

    response = requests.post(
        args.url,
        data=ormsgpack.packb(pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        stream=args.streaming,
        headers={
            "authorization": f"Bearer {args.api_key}",
            "content-type": "application/msgpack",
        },
    )

    if response.status_code == 200:
        if args.streaming:
            p = pyaudio.PyAudio()
            audio_format = pyaudio.paInt16  # Assuming 16-bit PCM format
            stream = p.open(
                format=audio_format, channels=args.channels, rate=args.rate, output=True
            )

            wf = wave.open(f"{args.output}.wav", "wb")
            wf.setnchannels(args.channels)
            wf.setsampwidth(p.get_sample_size(audio_format))
            wf.setframerate(args.rate)

            stream_stopped_flag = False

            try:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        stream.write(chunk)
                        wf.writeframesraw(chunk)
                    else:
                        if not stream_stopped_flag:
                            stream.stop_stream()
                            stream_stopped_flag = True
            finally:
                stream.close()
                p.terminate()
                wf.close()
        else:
            audio_content = response.content
            audio_path = f"{args.output}.{args.format}"
            with open(audio_path, "wb") as audio_file:
                audio_file.write(audio_content)

            audio = AudioSegment.from_file(audio_path, format=args.format)
            if args.play:
                play(audio)
            print(f"Audio has been saved to '{audio_path}'.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())
