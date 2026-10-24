# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
PACKAGE = SRC / "boltzmann_lab"
RESOURCE_ROOT = PACKAGE / "resources"

if not (PACKAGE / "__main__.py").is_file():
    raise FileNotFoundError(f"Application entry point not found: {PACKAGE / '__main__.py'}")

if not RESOURCE_ROOT.is_dir():
    raise FileNotFoundError(f"Application resources not found: {RESOURCE_ROOT}")

datas = [
    (str(path), str(path.parent.relative_to(SRC)))
    for path in RESOURCE_ROOT.rglob("*")
    if path.is_file()
]

a = Analysis(
    [str(PACKAGE / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BoltzmannLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
