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
| `meeting.py` | Módulo compartilhado: localiza a mídia na pasta da reunião e lê o `meeting.toml`. |

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

### 1. Prepare a pasta da reunião

Cada gravação ganha sua própria pasta em `input/`, com a mídia e um `meeting.toml` opcional:

```
input/
└── reuniao-planejamento/
    ├── reuniao-planejamento.mp4
    ├── meeting.toml        ← idioma, prompt, nº de participantes, anotações
    └── pauta.md            ← qualquer outro material de apoio (ignorado pelos scripts)
```

Copie [`meeting.example.toml`](meeting.example.toml) para a pasta e preencha o que souber:

```toml
language = "pt"        # sem este campo, o idioma é detectado automaticamente
speakers = 3           # sem este campo, é estimado automaticamente
prompt = """
Reunião de planejamento do projeto X com Ana, Bruno e Carla. Termos: Odoo, ERP, sprint.
"""
notes = """
Anotações livres (não usadas pelos scripts).
"""
```

O `prompt` é o principal ajuste de qualidade: nomes próprios, siglas e termos técnicos ali escritos tendem a sair grafados corretamente na transcrição. Argumentos de linha de comando têm prioridade sobre o `meeting.toml`.

### 2. Transcreva (fluxo recomendado, com interlocutores)

```bash
# Transcreve e faz a diarização inicial
python transcribe_diarized.py "input/reuniao-planejamento"

# Refina a diarização (encontra a transcrição automaticamente)
python refine_diarization.py "input/reuniao-planejamento"
```

Resultado em `output/reuniao-planejamento/` (o nome da subpasta é o nome do arquivo de mídia, sem extensão):

- `reuniao-planejamento.diarized.txt` — texto legível, com horário e interlocutor
- `reuniao-planejamento.diarized.srt` — legenda
- `reuniao-planejamento.diarized.json` — enunciados, turnos e segmentos de diarização

Opções do `transcribe_diarized.py`: `--model`, `--language`, `--prompt`, `--threads`, `--min-speakers`, `--max-speakers`, `--output-dir`.
Opções do `refine_diarization.py`: `--speakers`, `--transcript`, `--output-dir`.

> **Dica:** se só interessa uma voz principal (ex.: o professor numa aula), use `speakers = 2` para separar apenas "voz principal" e "demais".

### Transcrição simples

```bash
python transcribe.py "input/reuniao-planejamento"
```

Gera `.txt`, `.srt` e `.json` (sem interlocutores) na mesma pasta de saída. Mais rápido; usa `small.en` para inglês e `small` para outros idiomas, a menos que `--model` seja informado.

## Privacidade

Tudo roda localmente. As pastas `input/`, `output/` e `.models/` estão no `.gitignore` e nunca são versionadas.
