import os
from functions import parse_pdf_content, save_to_json

def main():
    pdf_path = os.path.abspath("data/raw_files/pdf/CvLatexEng.pdf")
    output_path = os.path.abspath("data/processed_files/JSON/text_content.json")
    
    print(f"Processing {pdf_path}...")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    try:
        content = parse_pdf_content(pdf_path)
        print(f"Extracted {len(content)} chunks.")
        
        save_to_json(content, output_path)
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
