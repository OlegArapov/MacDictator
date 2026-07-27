import sys
sys.setrecursionlimit(5000)
from setuptools import setup

APP = ['app.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'strip': False,  # strip fails on .so copied without owner-write bit
    'iconfile': 'assets/MacDictator.icns',
    'plist': {
        'CFBundleName': 'MacDictator',
        'CFBundleDisplayName': 'MacDictator',
        'CFBundleIdentifier': 'com.macdictator.app',
        'CFBundleVersion': '1.0.11',
        'CFBundleShortVersionString': '1.0.11',
        # нижняя граница задана колёсами mlx: Metal-кернелы собраны под этот SDK
        'LSMinimumSystemVersion': '14.0',
        'NSMicrophoneUsageDescription': 'MacDictator needs microphone access for speech-to-text.',
        'NSAppleEventsUsageDescription': 'MacDictator needs accessibility access to paste text.',
    },
    'packages': [
        # свои модули — папкой, а не внутри python312.zip: tray.py запускается
        # как отдельный скрипт по пути на диске
        'macdictator',
        'customtkinter',
        'openai',
        'sounddevice',
        'soundfile',
        # dylib payloads must live as real dirs, not inside python312.zip —
        # dlopen cannot load a library from a zip archive
        '_sounddevice_data',
        '_soundfile_data',
        'numpy',
        'pyperclip',
        'pyautogui',
        'pynput',
        'psutil',
        'mlx_whisper',
        'mlx',
    ],
    'includes': [
        'tkinter',
        '_tkinter',
        # AppKit импортируется лениво внутри функций (NSScreen/NSWorkspace/
        # NSApplication для баблов на всех мониторах и гейта Escape) — modulegraph
        # его сам не увидит, поэтому включаем явно.
        'AppKit',
        'Foundation',
        'objc',
        'Quartz',
    ],
    'frameworks': [],
    # torch приезжает зависимостью mlx-whisper, но внутри пакета его импортирует
    # только torch_whisper.py — конвертер оригинальных весов Whisper в формат MLX,
    # на который не ссылается ни один модуль (__init__ тянет audio, decoding,
    # load_models, transcribe). В транскрибации torch не участвует, а вместе с
    # sympy занимает половину бандла. scipy и numba не трогать: transcribe.py
    # импортирует timing.py на уровне модуля (issue #11).
    'excludes': [
        'torch',
        'torchgen',
        'sympy',
        'pygments',
    ],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
