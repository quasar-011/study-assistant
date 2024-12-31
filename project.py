import os
import requests
import re
import whisper
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from pydub.silence import detect_silence
import glob
import yt_dlp

# Download NLTK resources
nltk.download('punkt')

# Initialize BART model and tokenizer for summarization
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Set device to CPU or GPU based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize Whisper model for transcription
whisper_model = whisper.load_model("medium")

def fetch_book_content(url):
    """
    Fetches the content of the book from the given URL.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_title_and_author(content):
    """
    Extracts the title and author from the book's content.
    """
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

def create_directory(title, author):
    """
    Creates a directory named after the book's title and author.
    """
    invalid_chars = r'[<>:"/\\|?*\n]'
    clean_title = re.sub(invalid_chars, '', title)
    clean_author = re.sub(invalid_chars, '', author)
    dir_name = f"{clean_title} - {clean_author}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def save_chapters(content, dir_name):
    """
    Splits the book content into chapters and saves each chapter as a separate text file.
    """
    chapters = []
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find start or end markers for the book content.")
        return
    book_content = content[start_idx + len(start_marker):end_idx]
    book_content = book_content.replace('\r\n', '\n').replace('\r', '\n')
    chapter_pattern = re.compile(r'^(CHAPTER|Chapter|LETTER|Letter)\s+([IVXLCDM]+|\d+)', re.IGNORECASE | re.MULTILINE)
    matches = list(chapter_pattern.finditer(book_content))
    if not matches:
        print("Error: Could not find any chapters or letters in the book content.")
        return
    print(f"Total chapter/letter headings found: {len(matches)}")
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
            print(f"Extracted {chapter_title} with {word_count} words.")
        else:
            print(f"Skipped {chapter_title} due to insufficient content ({word_count} words).")
    for i, (title, chapter_text) in enumerate(chapters):
        safe_title = re.sub(r'[<>:"/\\|?*\n]', '', title.replace(' ', '_'))
        file_path = os.path.join(dir_name, f"Chapter_{i + 1}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n\n{chapter_text}")
        print(f"Saved {title} as {file_path}")

def summarize_text(text):
    """
    Summarizes the given text using the BART model.
    """
    sentences = sent_tokenize(text)
    summaries = []
    for i in range(0, len(sentences), 10):
        batch = ' '.join(sentences[i:i + 10])
        inputs = tokenizer(batch, return_tensors='pt', max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return ' '.join(summaries)

def download_youtube_audio(url):
    """
    Downloads audio from a YouTube video using yt-dlp.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "downloaded_audio.mp3"

def split_audio_on_silence(file_path, silence_thresh=-40, min_silence_len=700):
    """
    Splits audio into smaller chunks based on silence.
    """
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []
    silent_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    chunks = []
    start_time = 0
    for start, end in silent_ranges:
        if start_time < start:
            chunks.append(audio[start_time:start])
        start_time = end
    if start_time < len(audio):
        chunks.append(audio[start_time:])
    return chunks

def transcribe_audio_chunks(audio_chunks):
    """
    Transcribes a list of audio chunks using the Whisper model.
    """
    full_transcription = []
    for i, chunk in enumerate(audio_chunks):
        chunk.export(f"chunk_{i}.mp3", format="mp3")
        result = whisper_model.transcribe(f"chunk_{i}.mp3")
        full_transcription.append(result["text"])
        torch.cuda.empty_cache()
    return "\n".join(full_transcription)

def summarize_transcription(transcribed_text, batch_size=10):
    """
    Summarizes the transcription in batches of 10 sentences.
    """
    sentences = sent_tokenize(transcribed_text)
    summaries = []
    for i in range(0, len(sentences), batch_size):
        batch = ' '.join(sentences[i:i + batch_size])
        if batch.strip():
            try:
                inputs = tokenizer(batch, return_tensors='pt', max_length=1024, truncation=True).to(device)
                with torch.no_grad():
                    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory during summarization. Skipping this batch.")
    return ' '.join(summaries)

def book_summary_flow():
    """
    Handles book summarization flow.
    """
    url = input("Enter the link to the book: ").strip()
    try:
        content = fetch_book_content(url)
        print("Book content fetched successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the book: {e}")
        return
    title, author = extract_title_and_author(content)
    print(f"Title: {title}\nAuthor: {author}")
    dir_name = create_directory(title, author)
    save_chapters(content, dir_name)
    chapter_files = sorted(glob.glob(os.path.join(dir_name, "Chapter_*.txt")))
    num_chapters = len(chapter_files)
    if num_chapters == 0:
        print("No chapters were saved.")
        return
    print(f"Total chapters available for summarization: {num_chapters}")
    while True:
        chapter_choice = input(f"Which chapter to summarize (1-{num_chapters}) or 'b' to go back? ")
        if chapter_choice.lower() == 'b':
            break
        try:
            chapter_choice = int(chapter_choice)
            if 1 <= chapter_choice <= num_chapters:
                chapter_file_path = chapter_files[chapter_choice - 1]
                with open(chapter_file_path, "r", encoding="utf-8") as f:
                    chapter_text = f.read()
                summary = summarize_text(chapter_text)
                print(f"Chapter {chapter_choice} Summary:\n{summary}")
            else:
                print(f"Invalid choice. Please enter a number between 1 and {num_chapters}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def video_summary_flow():
    """
    Handles YouTube video summarization flow.
    """
    url = input("Enter the YouTube video link: ").strip()
    audio_file = download_youtube_audio(url)
    audio_chunks = split_audio_on_silence(audio_file)
    if not audio_chunks:
        print("No audio chunks were created. Exiting.")
        return
    full_transcribed_text = transcribe_audio_chunks(audio_chunks)
    summarized_text = summarize_transcription(full_transcribed_text)
    print(f"Summarized Text:\n{summarized_text}")

# Main interaction loop
while True:
    print("\nChoose an option:")
    print("1. Summarize a book")
    print("2. Summarize a YouTube video")
    print("3. Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        book_summary_flow()
    elif choice == "2":
        video_summary_flow()
    elif choice == "3":
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
