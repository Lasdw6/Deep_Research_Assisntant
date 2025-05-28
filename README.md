
# ScholarAI 🎓

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://huggingface.co/spaces/Lasdw/ScholarAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://huggingface.co/spaces/Lasdw/ScholarAI)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-5.29.1-orange)](https://gradio.app/)

An AI-powered research assistant that helps you find answers by searching the web, analyzing images, processing audio, and more.

## Features

- Web search and Wikipedia integration
- Image analysis
- Audio processing
- Code analysis
- Data file processing

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- gradio: Web interface
- langchain & langgraph: AI agent framework
- openai: Language model integration
- beautifulsoup4 & html2text: Web scraping
- pytube & youtube-transcript-api: Video processing
- whisper: Audio transcription
- pandas & openpyxl: Data processing
- Pillow: Image processing
- PyPDF2 & pymupdf: PDF handling

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Created by [Vividh Mahajan](https://huggingface.co/Lasdw)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---
title: ScholarAI
emoji: 🎓
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: true
hf_oauth: true
hf_oauth_expiration_minutes: 480
---
