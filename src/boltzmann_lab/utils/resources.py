import json
import os

from functools import lru_cache
from importlib.resources import files

from platformdirs import user_cache_dir
from pathlib import Path


PACKAGE = "boltzmann_lab.resources"


def configure_cache():
    """
    Настраивает пользовательский каталог кеша Numba.

    Вызывать до первого импорта numba.
    """
    cache_dir = user_cache_dir(
        appname="BoltzmannLab",
        appauthor=False,
    )

    numba_dir = os.path.join(
        cache_dir,
        "numba",
    )

    os.makedirs(
        numba_dir,
        exist_ok=True,
    )

    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        numba_dir,
    )


def load_text(relative_path):
    """
    Читает текстовый ресурс из пакета приложения.
    """
    return (
        files(PACKAGE)
        .joinpath(relative_path)
        .read_text(encoding="utf-8")
    )


def load_json(relative_path):
    """
    Читает JSON-ресурс.
    """
    return json.loads(
        load_text(relative_path)
    )


@lru_cache(maxsize=2)
def load_translation(language):
    """
    Загружает словарь переводов указанного языка.
    """
    if language not in ("ru", "en"):
        raise ValueError(
            f"Unsupported language: {language}"
        )

    return load_json(
        f"locales/{language}.json"
    )


@lru_cache(maxsize=6)
def load_document(language, name):
    """
    Загружает HTML-документ.
    """
    if language not in ("ru", "en"):
        raise ValueError(
            f"Unsupported language: {language}"
        )

    if name not in (
        "instructions",
        "theory",
        "authorship",
    ):
        raise ValueError(
            f"Unknown document: {name}"
        )

    return load_text(
        f"documents/{language}/{name}.html"
    )


@lru_cache(maxsize=1)
def load_stylesheet():
    """
    Загружает общий стиль Qt.
    """
    return load_text(
        "styles/main.qss"
    )

def resource_path(*parts):
    return Path(__file__).resolve().parent.parent / "resources" / Path(*parts)