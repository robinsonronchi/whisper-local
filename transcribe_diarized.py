import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from meeting import load_meeting


SAMPLE_RATE = 16_000


def timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{milliseconds:03}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcricao inglesa de alta fidelidade com diarizacao local."
    )
    parser.add_argument("input", type=Path, help="Pasta da reuniao (input/<nome>/) ou arquivo de midia.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", default=None, help="Padrao: large-v3-turbo.")
    parser.add_argument("--language", default=None, help="Ex.: en, pt. Padrao: deteccao automatica.")
    parser.add_argument("--prompt", default=None, help="Contexto, nomes proprios e termos tecnicos.")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--min-speakers", type=int, default=None, help="Padrao: 3.")
    parser.add_argument("--max-speakers", type=int, default=None, help="Padrao: 6.")
    return parser.parse_args()


def make_utterances(segments) -> list[dict]:
    utterances: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        nonlocal current
        if current:
            utterances.append(
                {
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": "".join(word["text"] for word in current).strip(),
                }
            )
            current = []

    for segment in segments:
        for word in segment.words or []:
            item = {"start": word.start, "end": word.end, "text": word.word}
            if current:
                pause = item["start"] - current[-1]["end"]
                duration = current[-1]["end"] - current[0]["start"]
                sentence_end = current[-1]["text"].rstrip().endswith((".", "?", "!"))
                if pause >= 0.65 or duration >= 12 or (sentence_end and duration >= 3):
                    flush()
            current.append(item)
        flush()
    return [item for item in utterances if item["text"]]


def voice_embedding(encoder: VoiceEncoder, audio: np.ndarray, start: float, end: float):
    left = max(0, int((start - 0.10) * SAMPLE_RATE))
    right = min(len(audio), int((end + 0.10) * SAMPLE_RATE))
    clip = audio[left:right].astype(np.float32)
    minimum = int(1.0 * SAMPLE_RATE)
    if len(clip) < minimum:
        clip = np.pad(clip, (0, minimum - len(clip)))
    peak = np.max(np.abs(clip)) if len(clip) else 0
    if peak > 0:
        clip = clip / peak * 0.95
    return encoder.embed_utterance(clip)


def cluster_speakers(embeddings: np.ndarray, minimum: int, maximum: int):
    maximum = min(maximum, len(embeddings) - 1)
    # silhouette exige ao menos 2 grupos; com maximum < 2 cai no caso de interlocutor unico
    minimum = max(2, min(minimum, maximum))
    best = None
    for count in range(minimum, maximum + 1):
        labels = AgglomerativeClustering(
            n_clusters=count, metric="cosine", linkage="average"
        ).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric="cosine")
        if best is None or score > best[0]:
            best = (score, labels, count)
    if best is None:
        return np.zeros(len(embeddings), dtype=int), 1, None
    return best[1], best[2], round(float(best[0]), 4)


def merge_turns(utterances: list[dict]) -> list[dict]:
    turns: list[dict] = []
    for utterance in utterances:
        if turns and turns[-1]["speaker"] == utterance["speaker"]:
            turns[-1]["end"] = utterance["end"]
            turns[-1]["text"] += " " + utterance["text"]
        else:
            turns.append(dict(utterance))
    return turns


def main() -> None:
    args = arguments()
    meeting = load_meeting(args.input)
    source = meeting.media
    destination = meeting.output_dir(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    model_name = meeting.option(args.model, "model", "large-v3-turbo")
    language = meeting.option(args.language, "language")
    prompt = meeting.option(args.prompt, "prompt")
    speakers = meeting.config.get("speakers")
    min_speakers = meeting.option(args.min_speakers, "min_speakers", speakers or 3)
    max_speakers = meeting.option(args.max_speakers, "max_speakers", speakers or 6)

    print(f"Transcrevendo {source.name} com {model_name} (CPU/int8)...", flush=True)
    model = WhisperModel(
        model_name, device="cpu", compute_type="int8", cpu_threads=args.threads
    )
    pipeline = BatchedInferencePipeline(model=model)
    segments_iterator, info = pipeline.transcribe(
        str(source),
        language=language,
        batch_size=8,
        beam_size=5,
        best_of=5,
        temperature=0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
        word_timestamps=True,
        initial_prompt=prompt,
    )
    segments = list(segments_iterator)
    utterances = make_utterances(segments)
    print(f"Transcricao concluida: {len(utterances)} enunciados.", flush=True)

    with tempfile.TemporaryDirectory(prefix="diarization-") as temp_dir:
        wav = Path(temp_dir) / "audio.wav"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav),
            ],
            check=True,
        )
        audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)

    print("Extraindo assinaturas vocais...", flush=True)
    encoder = VoiceEncoder(device="cpu")
    embeddings = np.vstack(
        [voice_embedding(encoder, audio, item["start"], item["end"]) for item in utterances]
    )
    labels, speaker_count, silhouette = cluster_speakers(
        embeddings, min_speakers, max_speakers
    )
    for utterance, label in zip(utterances, labels):
        utterance["speaker"] = f"SPEAKER_{label + 1:02d}"
    turns = merge_turns(utterances)

    stem = meeting.name
    txt_path = destination / f"{stem}.diarized.txt"
    srt_path = destination / f"{stem}.diarized.srt"
    json_path = destination / f"{stem}.diarized.json"

    with txt_path.open("w", encoding="utf-8") as output:
        for turn in turns:
            output.write(
                f"[{timestamp(turn['start'], ':')} - {timestamp(turn['end'], ':')}] "
                f"{turn['speaker']}: {turn['text']}\n\n"
            )
    with srt_path.open("w", encoding="utf-8") as output:
        for index, turn in enumerate(turns, 1):
            output.write(
                f"{index}\n{timestamp(turn['start'])} --> {timestamp(turn['end'])}\n"
                f"[{turn['speaker']}] {turn['text']}\n\n"
            )
    result = {
        "source": str(source),
        "model": model_name,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration_seconds": info.duration,
        "speaker_count": speaker_count,
        "speaker_labels": "Acoustic clusters; identities were not inferred.",
        "clustering_silhouette": silhouette,
        "utterances": utterances,
        "turns": turns,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Diarizacao concluida: {speaker_count} interlocutores estimados.")
    print(txt_path)
    print(srt_path)
    print(json_path)


if __name__ == "__main__":
    main()
