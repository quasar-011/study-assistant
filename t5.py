
import os
import requests
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.tokenize import sent_tokenize
import glob

# Download NLTK resources
nltk.download('punkt')

# Initialize the summarization pipeline with T5
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=0 if torch.cuda.is_available() else -1)

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

    # Look for title and author after the Gutenberg header
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
    # Remove characters that are invalid in file names
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
    
    # Define start and end markers
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find start or end markers for the book content.")
        return
    
    # Extract the main book content
    book_content = content[start_idx + len(start_marker):end_idx]
    
    # Normalize line endings
    book_content = book_content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Define a regex pattern to identify chapter and letter titles
    chapter_pattern = re.compile(r'^(CHAPTER|Chapter|LETTER|Letter)\s+([IVXLCDM]+|\d+)', re.IGNORECASE | re.MULTILINE)
    
    # Find all chapter and letter headings with their positions
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
        
        # Check if chapter has more than 100 words to ensure it's substantive
        word_count = len(chapter_content.split())
        if word_count > 100:
            chapters.append((chapter_title, chapter_content))
            print(f"Extracted {chapter_title} with {word_count} words.")
        else:
            print(f"Skipped {chapter_title} due to insufficient content ({word_count} words).")
    
    if not chapters:
        print("Error: No chapters with sufficient content were found.")
        return
    
    # Save chapters to separate text files
    for i, (title, chapter_text) in enumerate(chapters):
        # Replace spaces with underscores and remove invalid characters for filenames
        safe_title = re.sub(r'[<>:"/\\|?*\n]', '', title.replace(' ', '_'))
        file_name = f"{safe_title}.txt"
        file_path = os.path.join(dir_name, f"Chapter_{i + 1}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n\n{chapter_text}")
        print(f"Saved {title} as {file_path}")
    
    print(f"Total chapters saved: {len(chapters)}")

def summarize_text(text):
    """
    Summarizes the given text using the T5 model.
    Formats the summary into bullet points for better readability.
    """
    # Adjust parameters as needed
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    
    # Post-processing: Convert summary into bullet points
    # Here, we'll assume that sentences in the summary can be bullet points
    sentences = sent_tokenize(summary)
    bullet_points = '\n'.join([f"- {sentence}" for sentence in sentences])
    
    return bullet_points

def main():
    """
    Main function to execute the script.
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
    print(f"Directory created: {dir_name}")
    
    save_chapters(content, dir_name)
    
    # Use glob to find chapter files
    chapter_files = sorted(glob.glob(os.path.join(dir_name, "Chapter_*.txt")))
    num_chapters = len(chapter_files)
    
    if num_chapters == 0:
        print("No chapters were saved. Please check the content extraction logic.")
        return

    print(f"Total chapters available for summarization: {num_chapters}")
    
    while True:
        try:
            chapter_choice = int(input(f"Which chapter do you want summarized (1 to {num_chapters})? Enter 0 to exit: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        if chapter_choice == 0:
            print("Exiting program.")
            break

        if 1 <= chapter_choice <= num_chapters:
            chapter_file_path = chapter_files[chapter_choice - 1]
            with open(chapter_file_path, 'r', encoding='utf-8') as f:
                chapter_text = f.read()
            
            print(f"Summarizing Chapter {chapter_choice}...")
            summary = summarize_text(chapter_text)
            summary_file_path = os.path.join(dir_name, f"Chapter_{chapter_choice}_summary.txt")
            
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                f.write(summary)
    
            print(f"Summary of Chapter {chapter_choice} saved at {summary_file_path}")
            print(f"Summary of Chapter {chapter_choice}:\n{summary}\n")
        else:
            print(f"Chapter {chapter_choice} does not exist. Please choose a valid chapter number.")

if __name__ == "__main__":
    main()
