# Summarization App

## Overview

The **Summarization App** is a web-based tool that provides text summarization for both books and YouTube videos. The app can:
- Summarize chapters from books hosted on **Project Gutenberg** (only UTF-8 format).
- Generate summaries from **YouTube videos** using either existing transcripts or transcriptions of the audio.

This project leverages several state-of-the-art machine learning models, such as **BART** for text summarization and **Whisper** for audio transcription, all through an intuitive interface built with **Streamlit**.

## Features

### 1. **Book Summarization**:
- Fetches book content directly from Project Gutenberg (UTF-8 format only).
- Automatically extracts chapters from the book.
- Summarizes selected chapters using BART.
  
### 2. **YouTube Video Summarization**:
- Fetches and summarizes transcripts of YouTube videos.
- If no transcript is available, downloads and transcribes the audio using Whisper, then generates a summary.
  
## Requirements

### Python Version
Ensure you have Python 3.8 or above installed.

### Dependencies
The app relies on several key Python libraries:
- **streamlit**: For the web interface.
- **transformers**: To use BART for text summarization.
- **torch**: To handle model computations.
- **nltk**: For sentence tokenization.
- **whisper**: For audio transcription.
- **yt-dlp**: For downloading YouTube audio.
- **pytube**: To fetch video metadata.
- **youtube_transcript_api**: For fetching YouTube transcripts.
- **requests**: For fetching book content.
  
## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/summarization-app.git
cd summarization-app
```

### 2. Set up the Virtual Environment

It’s recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
The app requires the `punkt` tokenizer from NLTK for sentence tokenization. To download it:
```bash
python -c "import nltk; nltk.download('punkt')"
```

### 5. Install yt-dlp
Ensure that `yt-dlp` is installed to handle YouTube video downloads. You can install it via pip:
```bash
pip install yt-dlp
```

Or download directly from the official [yt-dlp repository](https://github.com/yt-dlp/yt-dlp).

## Running the App

To start the Streamlit app, run:
```bash
streamlit run app.py
```

This will open the app in your web browser.

## Usage

### Book Summarization

- **Supported Source**: The app only supports books from **Project Gutenberg** that are in **UTF-8 text format**.
- **Book URL Requirements**:
  - Ensure the book URL is from **Project Gutenberg**.
  - The book must be in **UTF-8 plain text format**.
  - The URL should directly point to the **UTF-8 version of the plain text** file.

#### How to Get the Right Format from Project Gutenberg:
1. Navigate to [Project Gutenberg](https://www.gutenberg.org).
2. Search for the book you wish to summarize.
3. On the book’s download page, scroll down to the **Download options**.
4. Click on the **Plain Text UTF-8** format.
   - Example: `https://www.gutenberg.org/files/1342/1342-0.txt` (for *Pride and Prejudice*).
5. Copy the URL of the UTF-8 plain text file and use it in the app.

If you attempt to use a different format (e.g., HTML, PDF, or other encodings), the app will not be able to process the book.

### YouTube Video Summarization

- **Supported Source**: The app accepts YouTube video URLs.
- **Transcript**: If the video has a transcript, the app fetches it automatically.
- **Transcription**: If no transcript is available, the app downloads the video’s audio and transcribes it using Whisper.

## App Interface

1. **Main Page**:
   - Select either **Book Summarization** or **YouTube Video Summarization**.
   
2. **Book Summarization**:
   - Enter the URL of a book from Project Gutenberg (in UTF-8 format).
   - Fetch the book content and select a chapter.
   - Summarize the selected chapter and download the summary.
   
3. **YouTube Video Summarization**:
   - Enter a YouTube video URL.
   - Fetch the video’s transcript (if available) or transcribe its audio using Whisper.
   - Summarize the transcript and download the summary.

## Models Used

### 1. **BART**:
- **Model**: `facebook/bart-large-cnn`
- **Purpose**: Summarization of text (used for both book chapters and YouTube transcripts).

### 2. **Whisper**:
- **Model**: `openai/whisper`
- **Purpose**: Transcription of YouTube audio (used when no transcript is available).

## Folder Structure

The summaries and transcripts are saved in the following structure:
```
summaries/
  └── <Book or Video Title> - <Author or Video ID>/
      ├── Chapter_<n>.txt
      ├── transcript.txt
      └── summary.txt
```

## Future Improvements

- Adding support for more book formats and sources.
- Enhancing the transcription quality using Whisper's larger models.
- Implementing a more robust chapter detection algorithm for books.

## License

This project is licensed under the MIT [License](https://github.com/quasar-011/study-assistant/blob/main/LICENSE.txt).
