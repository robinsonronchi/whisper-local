import argparse
import json
import subprocess
import tempfile
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import sherpa_onnx


def timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{milliseconds:03}"


def assign_speaker(item: dict, diarization: list[dict]) -> str:
    overlap = Counter()
    for segment in diarization:
        shared = max(0.0, min(item["end"], segment["end"]) - max(item["start"], segment["start"]))
        if shared:
            overlap[segment["speaker"]] += shared
    if overlap:
        return overlap.most_common(1)[0][0]
    midpoint = (item["start"] + item["end"]) / 2
    closest = min(
        diarization,
        key=lambda segment: min(abs(midpoint - segment["start"]), abs(midpoint - segment["end"])),
    )
    return closest["speaker"]


def merge(items: list[dict]) -> list[dict]:
    turns = []
    for item in items:
        if turns and turns[-1]["speaker"] == item["speaker"] and item["start"] - turns[-1]["end"] < 1.2:
            turns[-1]["end"] = item["end"]
            turns[-1]["text"] += " " + item["text"]
        else:
            turns.append(dict(item))
    return turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--speakers", type=int, default=4)
    args = parser.parse_args()

    source = args.input.resolve()
    transcript_path = args.transcript.resolve()
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    utterances = data["utterances"]
    root = Path(__file__).resolve().parent
    segmentation = root / ".models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
    embedding = root / ".models/diarization/nemo_en_titanet_small.onnx"

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(str(segmentation)),
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(str(embedding), num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=args.speakers),
        min_duration_on=0.3,
        min_duration_off=0.35,
    )
    if not config.validate():
        raise SystemExit("Configuracao de diarizacao invalida")

    with tempfile.TemporaryDirectory(prefix="sherpa-diarization-") as folder:
        wav_path = Path(folder) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
            check=True,
        )
        with wave.open(str(wav_path), "rb") as wav:
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)

    def progress(done: int, total: int) -> int:
        if done % 200 == 0 or done == total:
            print(f"Diarizacao: {done}/{total}", flush=True)
        return 0

    result = diarizer.process(audio, progress)
    segments = [
        {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": f"SPEAKER_{s.speaker + 1:02d}"}
        for s in result.sort_by_start_time()
    ]
    for item in utterances:
        item["speaker"] = assign_speaker(item, segments)
    turns = merge(utterances)

    data["speaker_count"] = result.num_speakers
    data["diarization_engine"] = "sherpa-onnx pyannote-segmentation-3.0 + NeMo TitaNet"
    data["speaker_labels"] = "Acoustic speaker clusters; personal identities were not inferred."
    data.pop("clustering_silhouette", None)
    data["diarization_segments"] = segments
    data["utterances"] = utterances
    data["turns"] = turns
    transcript_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    stem = transcript_path.name.removesuffix(".diarized.json")
    txt_path = transcript_path.with_name(stem + ".diarized.txt")
    srt_path = transcript_path.with_name(stem + ".diarized.srt")
    with txt_path.open("w", encoding="utf-8") as output:
        for turn in turns:
            output.write(f"[{timestamp(turn['start'], ':')} - {timestamp(turn['end'], ':')}] {turn['speaker']}: {turn['text']}\n\n")
    with srt_path.open("w", encoding="utf-8") as output:
        for index, turn in enumerate(turns, 1):
            output.write(f"{index}\n{timestamp(turn['start'])} --> {timestamp(turn['end'])}\n[{turn['speaker']}] {turn['text']}\n\n")
    print(f"Concluido: {result.num_speakers} interlocutores, {len(turns)} turnos.")


if __name__ == "__main__":
    main()
