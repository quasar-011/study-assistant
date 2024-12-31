# app.py
import streamlit as st
import os
import re
import requests
import subprocess
import glob

from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import nltk
from nltk.tokenize import sent_tokenize
from pytube import YouTube
import yt_dlp
import whisper

from youtube_transcript_api import YouTubeTranscriptApi

# ==================== Setup and Initialization ====================

# Set page configuration
st.set_page_config(page_title="Summarization App", layout="wide")

# Download NLTK resources
nltk.download('punkt')

# Initialize and cache the BART model
@st.cache_resource
def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

# Initialize and cache the Whisper model
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("small")  # Options: "base", "small", "medium", "large"
    return model

# Load models
bart_tokenizer, bart_model, device = load_bart_model()
whisper_model = load_whisper_model()

# ==================== Helper Functions ====================

# ----- Book Summarization Functions -----

def fetch_book_content(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_title_and_author(content):
    title = None
    author = None

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    metadata_section = content.split(start_marker, 1)[0]
    title_match = re.search(r"Title:\s*(.+)", metadata_section, re.IGNORECASE)
    author_match = re.search(r"Author:\s*(.+)", metadata_section, re.IGNORECASE)

    if title_match:
        title = title_match.group(1).strip()
    if author_match:
        author = author_match.group(1).strip()

    return title or "Unknown Title", author or "Unknown Author"

def create_directory(title, author):
    invalid_chars = r'[<>:"/\\|?*\n]'
    clean_title = re.sub(invalid_chars, '', title)
    clean_author = re.sub(invalid_chars, '', author)
    dir_name = f"{clean_title} - {clean_author}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

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

    if not chapters:
        st.error("No chapters with sufficient content were found.")
        return chapters

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

# ----- YouTube Summarization Functions -----

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    if "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    return None

def fetch_transcript(video_id):
    """
    Fetches the transcript of a YouTube video using YouTubeTranscriptApi.
    Returns the transcript text if available, else None.
    """
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
    """
    Downloads the audio of a YouTube video using yt-dlp.
    """
    audio_file_path = os.path.join(dir_name, "audio.mp3")
    try:
        command = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "-o", f"{dir_name}/audio.%(ext)s",
            video_url
        ]
        subprocess.run(command, check=True)
        if os.path.exists(audio_file_path):
            return audio_file_path
        else:
            st.error("Audio download failed.")
            return None
    except Exception as e:
        st.error(f"Error downloading audio with yt-dlp: {e}")
        return None

def generate_transcript_with_whisper(audio_file_path):
    """
    Transcribes audio to text using OpenAI's Whisper model.
    """
    try:
        result = whisper_model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

def summarize_in_batches(text, batch_size=10):
    """
    Summarizes text in batches of sentences to handle large texts.
    """
    sentences = sent_tokenize(text)
    batch_summaries = []

    for i in range(0, len(sentences), batch_size):
        batch = " ".join(sentences[i:i + batch_size])
        inputs = bart_tokenizer([batch], max_length=1024, return_tensors="pt", truncation=True).to(device)
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

    return "\n".join(batch_summaries)

# ==================== Streamlit App ====================

def main():
    st.title("📚📹 Summarization App")
    st.markdown("""
    Welcome to the **Summarization App**! Choose between summarizing a book or a YouTube video.
    """)

    # Sidebar for option selection
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose a task", ["Book Summarization", "YouTube Video Summarization"])

    base_dir = "summaries"
    os.makedirs(base_dir, exist_ok=True)

    # Initialize session state variables if they don't exist
    if option == "Book Summarization":
        if 'book_chapters' not in st.session_state:
            st.session_state['book_chapters'] = []
        if 'book_title' not in st.session_state:
            st.session_state['book_title'] = ""
        if 'book_author' not in st.session_state:
            st.session_state['book_author'] = ""
        if 'book_dir' not in st.session_state:
            st.session_state['book_dir'] = ""
        if 'book_summary' not in st.session_state:
            st.session_state['book_summary'] = ""

    if option == "YouTube Video Summarization":
        if 'video_summary' not in st.session_state:
            st.session_state['video_summary'] = ""

    # ----- Book Summarization -----
    if option == "Book Summarization":
        st.header("📖 Book Summarization")
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
                    dir_name = create_directory(title, author)
                    chapters = save_chapters(content, dir_name)
                    if chapters:
                        st.session_state['book_chapters'] = chapters
                        st.session_state['book_title'] = title
                        st.session_state['book_author'] = author
                        st.session_state['book_dir'] = dir_name
                        st.success(f"Chapters saved in `{dir_name}`.")
                    else:
                        st.error("No chapters were saved. Please check the content extraction logic.")

        # Display chapter selection and summarization if chapters are available
        if st.session_state['book_chapters']:
            st.subheader(f"📄 Select a Chapter from '{st.session_state['book_title']}' by {st.session_state['book_author']}")

            chapter_options = [f"{i+1}. {chap[0]}" for i, chap in enumerate(st.session_state['book_chapters'])]
            selected_chapter = st.selectbox("Choose a chapter to summarize:", chapter_options)

            if st.button("Summarize Selected Chapter"):
                chapter_index = chapter_options.index(selected_chapter)
                chapter_title, chapter_text = st.session_state['book_chapters'][chapter_index]

                with st.spinner("Generating summary..."):
                    summary = summarize_text(chapter_text)
                    st.session_state['book_summary'] = summary
                    st.success("Summary generated successfully.")

                st.subheader("📝 Summary")
                st.write(summary)

                # Option to download the summary
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"{chapter_title}_summary.txt",
                    mime="text/plain"
                )

    # ----- YouTube Video Summarization -----
    if option == "YouTube Video Summarization":
        st.header("🎥 YouTube Video Summarization")
        video_url = st.text_input("Enter the YouTube video URL:")

        if st.button("Fetch and Summarize Video"):
            if not video_url:
                st.error("Please enter a valid YouTube URL.")
            else:
                video_id = extract_video_id(video_url)
                if not video_id:
                    st.error("Invalid YouTube URL.")
                else:
                    # Attempt to get video title using pytube
                    try:
                        yt = YouTube(video_url)
                        video_title = yt.title
                    except Exception as e:
                        st.warning("Could not fetch video title. Using video ID as title.")
                        video_title = f"Video_{video_id}"

                    dir_name = create_directory(video_title, "")

                    with st.spinner("Fetching transcript..."):
                        transcript = fetch_transcript(video_id)

                    if transcript:
                        st.success("Human-generated transcript found.")
                        with open(os.path.join(dir_name, "transcript.txt"), 'w', encoding='utf-8') as f:
                            f.write(transcript)
                        st.write("Transcript saved.")

                        with st.spinner("Generating summary..."):
                            summary = summarize_in_batches(transcript)
                            st.session_state['video_summary'] = summary
                            with open(os.path.join(dir_name, "summary.txt"), 'w', encoding='utf-8') as f:
                                f.write(summary)
                            st.success("Summary generated successfully.")

                        st.subheader("📝 Summary")
                        st.write(summary)

                        # Option to download the summary
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name="YouTube_Summary.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No human-generated transcript found. Proceeding to download audio and transcribe.")
                        with st.spinner("Downloading audio..."):
                            audio_file_path = download_audio_with_ytdlp(video_url, dir_name)

                        if audio_file_path:
                            with st.spinner("Transcribing audio..."):
                                transcript = generate_transcript_with_whisper(audio_file_path)
                                if transcript:
                                    with open(os.path.join(dir_name, "transcript.txt"), 'w', encoding='utf-8') as f:
                                        f.write(transcript)
                                    st.success("Transcription completed.")

                                    with st.spinner("Generating summary..."):
                                        summary = summarize_in_batches(transcript)
                                        st.session_state['video_summary'] = summary
                                        with open(os.path.join(dir_name, "summary.txt"), 'w', encoding='utf-8') as f:
                                            f.write(summary)
                                        st.success("Summary generated successfully.")

                                    st.subheader("📝 Summary")
                                    st.write(summary)

                                    # Option to download the summary
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
