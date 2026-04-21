import sys
sys.setrecursionlimit(5000)
from setuptools import setup

APP = ['app.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'MacDictator.icns',
    'plist': {
        'CFBundleName': 'MacDictator',
        'CFBundleDisplayName': 'MacDictator',
        'CFBundleIdentifier': 'com.macdictator.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSMicrophoneUsageDescription': 'MacDictator needs microphone access for speech-to-text.',
        'NSAppleEventsUsageDescription': 'MacDictator needs accessibility access to paste text.',
    },
    'packages': [
        'customtkinter',
        'openai',
        'sounddevice',
        'soundfile',
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
    ],
    'frameworks': [],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
