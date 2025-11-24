import json
import os

json_path = 'data/processed_files/JSON/text_content.json'

try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} chunks.")
    
    print("\n--- ALL CHUNKS ---")
    for i, item in enumerate(data):
        print(f"Chunk {i+1}:")
        print(f"Title: {item.get('snippet_title', 'N/A')}")
        print(f"Section: {item.get('section', 'N/A')}")
        print(f"Text: {item['text']}") 
        print("-" * 20)

except Exception as e:
    print(f"Error reading JSON: {e}")
