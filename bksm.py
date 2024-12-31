import streamlit as st  # Must be first Streamlit import
import os
import re
import requests
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

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

bart_tokenizer, bart_model, device = load_bart_model()

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

# Initialize session state for chapters and chapter selection
if 'chapters' not in st.session_state:
    st.session_state['chapters'] = None
if 'selected_chapter' not in st.session_state:
    st.session_state['selected_chapter'] = None

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
                        st.session_state['chapters'] = chapters  # Save chapters in session state
                        st.success(f"Chapters saved in {dir_name}.")
                    else:
                        st.error("No chapters were saved. Please check the content extraction logic.")

    # Show the selectbox for chapters if they have been fetched
    if st.session_state['chapters']:
        chapters = st.session_state['chapters']
        chapter_options = [f"{i+1}. {chap[0]}" for i, chap in enumerate(chapters)]
        chapter_choice = st.selectbox("Select a chapter to summarize:", chapter_options)

        if st.button("Summarize Chapter"):
            st.session_state['selected_chapter'] = chapter_choice  # Save selected chapter in session state
            index = chapter_options.index(chapter_choice)
            chapter_text = chapters[index][1]

            with st.spinner("Generating summary..."):
                summary = summarize_text(chapter_text)
                st.subheader("Summary")
                st.write(summary)
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"{chapters[index][0]}_summary.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
