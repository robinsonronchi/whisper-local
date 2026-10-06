import tomllib
from dataclasses import dataclass, field
from pathlib import Path


CONFIG_NAME = "meeting.toml"
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi",
    ".m4a", ".mp3", ".wav", ".ogg", ".opus", ".flac", ".aac", ".wma",
}


@dataclass
class Meeting:
    name: str
    media: Path
    folder: Path
    config: dict = field(default_factory=dict)

    def option(self, cli_value, key: str, default=None):
        """Prioridade: argumento de linha de comando > meeting.toml > padrao."""
        if cli_value is not None:
            return cli_value
        value = self.config.get(key)
        if isinstance(value, str):
            value = value.strip() or None
        return default if value is None else value

    def output_dir(self, base: Path) -> Path:
        return base.expanduser().resolve() / self.name


def load_meeting(path: Path) -> Meeting:
    """Aceita a pasta da reuniao (input/<nome>/) ou o arquivo de midia diretamente."""
    path = path.expanduser().resolve()
    if path.is_dir():
        media_files = sorted(
            item for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS
        )
        if not media_files:
            raise SystemExit(f"Nenhuma midia encontrada em: {path}")
        if len(media_files) > 1:
            names = "\n  ".join(item.name for item in media_files)
            raise SystemExit(
                f"Mais de uma midia em {path}; informe o arquivo desejado:\n  {names}"
            )
        media = media_files[0]
    elif path.is_file():
        media = path
    else:
        raise SystemExit(f"Arquivo ou pasta nao encontrado: {path}")

    folder = media.parent
    config_path = folder / CONFIG_NAME
    config = {}
    if config_path.is_file():
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        print(f"Configuracao: {config_path}", flush=True)

    return Meeting(name=media.stem, media=media, folder=folder, config=config)
