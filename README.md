# whisper-local

Transcrição local de reuniões (áudio ou vídeo) com identificação de interlocutores, rodando 100% na CPU — nenhum arquivo sai da sua máquina.

Usa [faster-whisper](https://github.com/SYSTRAN/faster-whisper) para transcrever e [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) para a diarização (quem falou quando).

## Exemplo de saída

```
[00:00:00:000 - 00:00:28:420] SPEAKER_01: As I already told you, it's integrated with the point of sale...

[00:00:46:330 - 00:01:06:830] SPEAKER_02: I don't know, just an idea. This is not the point where the nurse...
```

Os interlocutores são agrupamentos acústicos (`SPEAKER_01`, `SPEAKER_02`, …); nenhuma identidade é inferida.

## Scripts

| Script | O que faz |
| --- | --- |
| `transcribe.py` | Transcrição simples e rápida, sem separar interlocutores. Configurável (modelo, idioma, prompt). |
| `transcribe_diarized.py` | Transcrição de alta fidelidade (`large-v3-turbo`) com timestamps por palavra e uma primeira diarização via Resemblyzer + clustering. |
| `refine_diarization.py` | Refina a diarização de uma transcrição já gerada usando pyannote-segmentation-3.0 + NeMo TitaNet (sherpa-onnx). Mais preciso; recomendado após o passo anterior. |

## Requisitos

- Python 3.12
- `ffmpeg` no PATH
- ~4 GB de RAM livres para o `large-v3-turbo`

## Instalação

```bash
git clone https://github.com/robinsonronchi/whisper-local.git
cd whisper-local
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Os modelos Whisper são baixados automaticamente na primeira execução. Os modelos de diarização (usados só pelo `refine_diarization.py`) precisam ser baixados uma vez:

```bash
mkdir -p .models/diarization && cd .models/diarization
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx
cd ../..
```

## Uso

Coloque os arquivos em `input/`. Cada transcrição é gravada em `output/<nome do arquivo sem extensão>/`.

### Fluxo recomendado (com interlocutores)

```bash
# 1. Transcreve e faz a diarização inicial
python transcribe_diarized.py "input/reuniao.mp4"

# 2. Refina a diarização (informe o número de participantes, se souber)
python refine_diarization.py "input/reuniao.mp4" "output/reuniao/reuniao.diarized.json" --speakers 4
```

Resultado em `output/reuniao/`:

- `reuniao.diarized.txt` — texto legível, com horário e interlocutor
- `reuniao.diarized.srt` — legenda
- `reuniao.diarized.json` — enunciados, turnos e segmentos de diarização

Opções do `transcribe_diarized.py`: `--model`, `--threads`, `--min-speakers`, `--max-speakers`, `--output-dir`.

> O idioma está fixo em inglês e há um `initial_prompt` com vocabulário de domínio em `transcribe_diarized.py`. Ajuste ambos para o seu caso.

### Transcrição simples

```bash
python transcribe.py "input/reuniao.mp4" --model small.en --language en
```

Gera `reuniao.txt`, `reuniao.srt` e `reuniao.json` em `output/reuniao/`. Use `--prompt` para passar nomes próprios ou termos técnicos e `--model medium` / `--language pt` para outros idiomas.

## Privacidade

Tudo roda localmente. As pastas `input/`, `output/` e `.models/` estão no `.gitignore` e nunca são versionadas.
