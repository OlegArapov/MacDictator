import sys
sys.setrecursionlimit(5000)
from setuptools import setup

APP = ['app.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'strip': False,  # strip fails on .so copied without owner-write bit
    'iconfile': 'MacDictator.icns',
    'plist': {
        'CFBundleName': 'MacDictator',
        'CFBundleDisplayName': 'MacDictator',
        'CFBundleIdentifier': 'com.macdictator.app',
        'CFBundleVersion': '1.0.7',
        'CFBundleShortVersionString': '1.0.7',
        'NSMicrophoneUsageDescription': 'MacDictator needs microphone access for speech-to-text.',
        'NSAppleEventsUsageDescription': 'MacDictator needs accessibility access to paste text.',
    },
    'packages': [
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
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
