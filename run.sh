#!/bin/bash
# Runs nexus.py with the correct Python install.
#
# Installing Ollama via Homebrew pulled in its own Python (3.14) which now
# shadows the actual project Python (3.12, with opencv/whisper/torch/etc
# already installed) when you type plain `python3`. Rather than change your
# shell's global PATH, this script just points directly at the right one.
cd "$(dirname "$0")"
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 nexus.py
