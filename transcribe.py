import argparse
import json
import os
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

from meeting import load_meeting


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcreve áudio ou vídeo localmente com faster-whisper."
    )

    parser.add_argument(
        "input",
        help="Pasta da reunião (input/<nome>/) ou arquivo MP4, MP3, WAV etc.",
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Diretório base; os resultados vão para uma subpasta com o nome do arquivo.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Modelo Whisper. Padrão: small.en para inglês, small para outros idiomas.",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Idioma falado no arquivo. Padrão: en.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Quantidade de threads de CPU.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamanho do lote de processamento.",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Quantidade de alternativas consideradas durante a decodificação.",
    )

    parser.add_argument(
        "--prompt",
        default=None,
        help="Termos, nomes próprios ou contexto para orientar a transcrição.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    meeting = load_meeting(Path(args.input))
    input_path = meeting.media
    base_name = meeting.name

    args.language = meeting.option(args.language, "language", "en")
    args.model = meeting.option(
        args.model, "model", "small.en" if args.language == "en" else "small"
    )
    args.prompt = meeting.option(args.prompt, "prompt")

    output_dir = meeting.output_dir(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"
    json_path = output_dir / f"{base_name}.json"

    print(f"Arquivo: {input_path}")
    print(f"Modelo: {args.model}")
    print(f"Idioma: {args.language}")
    print(f"Threads: {args.threads}")
    print(f"Batch size: {args.batch_size}")
    print("Carregando modelo...")

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        cpu_threads=args.threads,
    )

    pipeline = BatchedInferencePipeline(model=model)

    segments, info = pipeline.transcribe(
        str(input_path),
        language=args.language,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        vad_filter=True,
        word_timestamps=False,
        initial_prompt=args.prompt,
    )

    json_segments = []
    subtitle_index = 1

    print("Transcrevendo...")

    with (
        txt_path.open("w", encoding="utf-8") as txt_file,
        srt_path.open("w", encoding="utf-8") as srt_file,
    ):
        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            txt_file.write(text + "\n")

            srt_file.write(f"{subtitle_index}\n")
            srt_file.write(
                f"{format_srt_timestamp(segment.start)} --> "
                f"{format_srt_timestamp(segment.end)}\n"
            )
            srt_file.write(text + "\n\n")

            json_segments.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": text,
                }
            )

            duration = getattr(info, "duration", 0)

            if duration:
                progress = min(100, segment.end / duration * 100)
                print(
                    f"\rProgresso aproximado: {progress:6.2f}%",
                    end="",
                    flush=True,
                )

            subtitle_index += 1

    result = {
        "source": str(input_path),
        "model": args.model,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration_seconds": getattr(info, "duration", None),
        "segments": json_segments,
    }

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=2)

    print("\n\nTranscrição concluída.")
    print(f"TXT:  {txt_path}")
    print(f"SRT:  {srt_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
