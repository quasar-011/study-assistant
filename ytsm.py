# app.py
import streamlit as st  # Must be first Streamlit import
import os
import re
import glob
import subprocess
import shutil
from io import StringIO

import requests
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

import nltk
from nltk.tokenize import sent_tokenize

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

import whisper  # Ensure correct whisper library is installed

# Set page config as the first Streamlit command
st.set_page_config(page_title="Summarization App", layout="wide")

# Initialize NLTK resources
nltk.download('punkt')

# Initialize Models with caching
@st.cache_resource
def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("small")  # Ensure using openai-whisper
    return model

bart_tokenizer, bart_model, device = load_bart_model()
whisper_model = load_whisper_model()

# Helper Functions for Book Summarization
def fetch_book_content(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_title_and_author(content):
    title = None
    author = None

    metadata_section = content.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    title_match = re.search(r"Title:\s*(.+)", metadata_section, re.IGNORECASE)
    author_match = re.search(r"Author:\s*(.+)", metadata_section, re.IGNORECASE)

    if title_match:
        title = title_match.group(1).strip()
    if author_match:
        author = author_match.group(1).strip()

    return title or "Unknown Title", author or "Unknown Author"

def create_directory(base_dir, title, author):
    invalid_chars = r'[<>:"/\\|?*\n]'
    clean_title = re.sub(invalid_chars, '', title)
    clean_author = re.sub(invalid_chars, '', author)
    dir_name = f"{clean_title} - {clean_author}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def save_chapters(content, dir_name):
    chapters = []

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        st.error("Could not find the start or end markers for the book content.")
        return chapters

    book_content = content[start_idx + len(start_marker):end_idx]
    book_content = book_content.replace('\r\n', '\n').replace('\r', '\n')

    chapter_pattern = re.compile(r'^(CHAPTER|Chapter|LETTER|Letter)\s+([IVXLCDM]+|\d+)', re.IGNORECASE | re.MULTILINE)
    matches = list(chapter_pattern.finditer(book_content))

    if not matches:
        st.error("No chapters or letters found in the book content.")
        return chapters

    for i, match in enumerate(matches):
        chapter_type = match.group(1).capitalize()
        chapter_number = match.group(2)
        chapter_title = f"{chapter_type} {chapter_number}"
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(book_content)

        chapter_content = book_content[start:end].strip()
        word_count = len(chapter_content.split())
        if word_count > 100:
            chapters.append((chapter_title, chapter_content))

    for i, (title, chapter_text) in enumerate(chapters):
        safe_title = re.sub(r'[<>:"/\\|?*\n]', '', title.replace(' ', '_'))
        file_path = os.path.join(dir_name, f"Chapter_{i + 1}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n\n{chapter_text}")

    return chapters

def summarize_text(text):
    sentences = sent_tokenize(text)
    summaries = []

    for i in range(0, len(sentences), 10):
        batch = ' '.join(sentences[i:i + 10])
        inputs = bart_tokenizer(batch, return_tensors='pt', max_length=1024, truncation=True).to(device)

        with torch.no_grad():
            summary_ids = bart_model.generate(
                inputs['input_ids'],
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return ' '.join(summaries)

# Helper Functions for YouTube Summarization
def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    if "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    return None

def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        for transcript in transcript_list:
            if not transcript.is_generated:
                transcript_data = transcript.fetch()
                text = "\n".join([entry["text"] for entry in transcript_data])
                return text

        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def download_audio_with_ytdlp(video_url, dir_name):
    try:
        command = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "-o", f"{dir_name}/audio.%(ext)s",
            video_url
        ]
        subprocess.run(command, check=True)
        audio_file_path = os.path.join(dir_name, "audio.mp3")
        if os.path.exists(audio_file_path):
            return audio_file_path
        else:
            st.error("Audio download failed.")
            return None
    except Exception as e:
        st.error(f"Error downloading audio with yt-dlp: {e}")
        return None

def summarize_in_batches(text, batch_size=10):
    sentences = sent_tokenize(text)
    batch_summaries = []

    for i in range(0, len(sentences), batch_size):
        batch = " ".join(sentences[i:i + batch_size])
        inputs = bart_tokenizer([batch], max_length=1024, return_tensors="pt", truncation=True)
        inputs = inputs.to(device)
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        batch_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        batch_summaries.append(batch_summary)

    full_summary = "\n".join(batch_summaries)
    return full_summary

def transcribe_audio(audio_file_path):
    try:
        result = whisper_model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

# Streamlit App
def main():
    st.title("📚📹 Summarization App")
    st.markdown("""
    This app allows you to summarize books and YouTube videos using advanced NLP models.
    """)

    option = st.selectbox(
        "Choose an option:",
        ("Summarize a Book", "Summarize a YouTube Video")
    )

    base_dir = "summaries"
    os.makedirs(base_dir, exist_ok=True)

    if option == "Summarize a Book":
        st.header("Book Summarization")
        book_url = st.text_input("Enter the URL of the book (e.g., Project Gutenberg link):")

        if st.button("Fetch and Process Book"):
            if not book_url:
                st.error("Please enter a valid book URL.")
            else:
                with st.spinner("Fetching book content..."):
                    try:
                        content = fetch_book_content(book_url)
                        st.success("Book content fetched successfully.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error fetching the book: {e}")
                        return

                with st.spinner("Extracting title and author..."):
                    title, author = extract_title_and_author(content)
                    st.write(f"**Title:** {title}")
                    st.write(f"**Author:** {author}")

                with st.spinner("Saving chapters..."):
                    dir_name = create_directory(base_dir, title, author)
                    chapters = save_chapters(content, dir_name)
                    if chapters:
                        st.success(f"Chapters saved in {dir_name}.")
                        chapter_options = [f"{i+1}. {chap[0]}" for i, chap in enumerate(chapters)]
                        chapter_choice = st.selectbox("Select a chapter to summarize:", chapter_options)
                        if st.button("Summarize Chapter"):
                            index = chapter_options.index(chapter_choice)
                            chapter_text = chapters[index][1]
                            with st.spinner("Generating summary..."):
                                summary = summarize_text(chapter_text)
                                st.subheader("Summary")
                                st.write(summary)
                                # Optionally, allow downloading the summary
                                st.download_button(
                                    label="Download Summary",
                                    data=summary,
                                    file_name=f"{chapters[index][0]}_summary.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.error("No chapters were saved. Please check the content extraction logic.")

    elif option == "Summarize a YouTube Video":
        st.header("YouTube Video Summarization")
        youtube_url = st.text_input("Enter the YouTube video URL:")

        if st.button("Fetch and Summarize Video"):
            if not youtube_url:
                st.error("Please enter a valid YouTube URL.")
            else:
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    st.error("Invalid YouTube URL.")
                else:
                    video_title = f"Video_{video_id}"
                    dir_name = create_directory(base_dir, video_title, "")

                    with st.spinner("Fetching transcript..."):
                        transcript = fetch_transcript(video_id)

                    if transcript:
                        st.success("Human-generated transcript found.")
                        save_path = os.path.join(dir_name, "transcript.txt")
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(transcript)
                        st.write("Transcript saved.")
                        with st.spinner("Generating summary..."):
                            summary = summarize_in_batches(transcript)
                            summary_path = os.path.join(dir_name, "summary.txt")
                            with open(summary_path, 'w', encoding='utf-8') as f:
                                f.write(summary)
                            st.subheader("Summary")
                            st.write(summary)
                            st.download_button(
                                label="Download Summary",
                                data=summary,
                                file_name="YouTube_Summary.txt",
                                mime="text/plain"
                            )
                    else:
                        st.warning("No human-generated transcript found. Proceeding to download audio and transcribe.")
                        with st.spinner("Downloading audio..."):
                            audio_path = download_audio_with_ytdlp(youtube_url, dir_name)

                        if audio_path:
                            with st.spinner("Transcribing audio..."):
                                transcript = transcribe_audio(audio_path)
                                if transcript:
                                    transcript_path = os.path.join(dir_name, "transcript.txt")
                                    with open(transcript_path, 'w', encoding='utf-8') as f:
                                        f.write(transcript)
                                    st.success("Transcription completed.")
                                    with st.spinner("Generating summary..."):
                                        summary = summarize_in_batches(transcript)
                                        summary_path = os.path.join(dir_name, "summary.txt")
                                        with open(summary_path, 'w', encoding='utf-8') as f:
                                            f.write(summary)
                                        st.subheader("Summary")
                                        st.write(summary)
                                        st.download_button(
                                            label="Download Summary",
                                            data=summary,
                                            file_name="YouTube_Summary.txt",
                                            mime="text/plain"
                                        )
                                else:
                                    st.error("Transcription failed.")
                        else:
                            st.error("Audio download failed.")

if __name__ == "__main__":
    main()
