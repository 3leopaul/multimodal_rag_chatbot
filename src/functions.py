import fitz
import os
import json
import re
from transformers import CLIPProcessor, CLIPTextModelWithProjection
import torch

def pre_process_text(text):
    """
    Initial cleaning: fixes hyphenation and removes artifacts, but PRESERVES newlines.
    """
    # Fix hyphenation (e.g., "Da-\ntabases" -> "Databases")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove "read more" artifacts
    text = re.sub(r'read\s*more', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.\.\.', '.', text)
    
    # Fix encoding replacement characters
    text = text.replace('', '')
    
    return text

def post_process_text(text):
    """
    Final cleaning: normalizes whitespace for chunking.
    """
    # Normalize whitespace (flattens text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Split text into chunks with overlap.
def split_text_with_overlap(text, max_length=256, overlap_percentage=0.25):
    # ... (same as before)
    if len(text) <= max_length:
        return [text]
    
    overlap_size = int(max_length * overlap_percentage)
    chunks = []
    
    start = 0
    while start < len(text):
        end = start + max_length
        
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Try to find a sentence ending to break at
        last_period = text.rfind('.', start, end)
        if last_period != -1 and last_period > start + max_length // 2:
            end = last_period + 1
        else:
            # Fallback to last space
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap_size
        # Ensure we move forward
        if start <= 0:
             start = end 
    
    return chunks

def get_section_header(chunk):
    # ... (kept for compatibility but less used now)
    MAJOR_HEADERS = {
        "EDUCATION": "EDUCATION",
        "SKILLS": "SKILLS",
        "PROFESSIONAL EXPERIENCE": "PROFESSIONAL EXPERIENCE",
        "LANGUAGES": "LANGUAGES",
        "INTERESTS": "INTERESTS",
        "PROJECTS": "PROJECTS",
        "SUMMARY": "SUMMARY"
    }
    
    for pattern, title in MAJOR_HEADERS.items():
        if re.search(r'(?:^|\n)\s*' + re.escape(pattern) + r'[:\s]*$', chunk[:50], re.IGNORECASE):
            return title
            
    first_line = chunk.split('\n')[0].strip()
    if len(first_line) > 5 and len(first_line) < 60 and first_line.isupper() and not first_line in MAJOR_HEADERS.values():
         return first_line
         
    if 'Engineering Student' in chunk[:100]:
        return "Summary & Contact"
        
    return None

def parse_pdf_content(pdf_path):
    # ... (existing setup code)
    doc = fitz.open(pdf_path)
    article_title_base = os.path.basename(pdf_path)
    structured_content = []
    
    MAJOR_HEADERS = [
        "EDUCATION", "SKILLS", "PROFESSIONAL EXPERIENCE", "LANGUAGES", 
        "INTERESTS", "PROJECTS", "SUMMARY"
    ]
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        
        # 1. Pre-process (keep newlines for regex)
        text = pre_process_text(text)
        
        if not text.strip():
            continue
            
        # 2. Split text by major headers
        pattern = r'(?=(?:^|\n)\s*(?:' + '|'.join(map(re.escape, MAJOR_HEADERS)) + r')[:\s])'
        sections = re.split(pattern, text, flags=re.IGNORECASE)
        
        current_header = "Summary & Contact"
        
        for section_text in sections:
            if not section_text.strip():
                continue
                
            # Check if this section starts with a header
            header_match = re.match(r'(?:^|\n)\s*(' + '|'.join(map(re.escape, MAJOR_HEADERS)) + r')[:\s]', section_text, re.IGNORECASE)
            if header_match:
                current_header = header_match.group(1).upper()
                # Remove the header from the text to avoid repetition if desired, 
                # or keep it. Let's keep it but ensure it's clean.
            
            # 3. Post-process (flatten)
            cleaned_section_text = post_process_text(section_text)
            
            # 4. Chunk
            section_chunks = split_text_with_overlap(cleaned_section_text)
            
            for chunk in section_chunks:
                structured_content.append({
                    'snippet_title': f"{article_title_base}: {current_header}",
                    'section': current_header,
                    'text': chunk
                })
    
    return structured_content

# Parse PDF content and extract images.

def parse_pdf_images(pdf_path):

    # Open the PDF file using PyMuPDF
    doc = fitz.open(pdf_path)
    # Extract the filename as the snippet title
    snippet_title = os.path.basename(pdf_path)
    # Initialize the list to store structured image data
    structured_content = []
    

    
    # Iterate through each page of the PDF
    for page_num, page in enumerate(doc):
        # Get a list of all images on the current page
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            # Extract the image reference number
            xref = img[0]
            # Extract the image dictionary including binary data and extension
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Construct a unique filename using title, page number, and image index
            image_filename = f"{os.path.splitext(snippet_title)[0]}_p{page_num+1}_{img_index}.{image_ext}"
            # Sanitize the filename by removing invalid characters
            image_filename = re.sub(r'[<>:"/\\\\|?*]', '_', image_filename).strip('_')
            # Define the full local path for saving the extracted image
            local_image_path = os.path.join('../data/processed_files/extracted_images/', image_filename)
            
            # Write the image binary data to the specified file path
            with open(local_image_path, "wb") as f:
                f.write(image_bytes)
            
            # Append the image metadata and path to the structured content list
            structured_content.append({
                'snippet_title': snippet_title,
                'section': f"Page {page_num + 1}",
                'image_path': local_image_path,
                # Add a descriptive caption including the source and page number
                'caption': f"Image from {snippet_title}, Page {page_num + 1}"
            })
            
    return structured_content

# Save structured content to a JSON file.
def save_to_json(structured_content, output_file='output.json'):
    """
    
    
    Args:
        structured_content (list): List of dictionaries containing structured content
        output_file (str): Path to the output JSON file (default: 'output.json')
    """
    # Create the output directory if it does not already exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save the structured content list to a JSON file with indentation
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_content, f, indent=4, ensure_ascii=False)

# Load structured content from a JSON file.
def load_from_json(input_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Global model and processor cache (loaded once)
_CLIP_MODEL = None
_CLIP_PROCESSOR = None

def _get_clip_model():
    """Lazy-load and cache the CLIP model and processor."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    
    if _CLIP_MODEL is None:
        print("Loading CLIP model (one-time initialization)...")
        _CLIP_MODEL = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        print("CLIP model loaded successfully!")
    
    return _CLIP_MODEL, _CLIP_PROCESSOR

# Convert text to embeddings using CLIP
def embed_text(text):
    """
    Generate text embeddings using CLIP model.
    Model is cached after first load for performance.
    """
    # Get cached model and processor
    model, processor = _get_clip_model()
    
    # Tokenize and preprocess the input text to prepare it for the model
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    # Pass the processed inputs through the model to generate text embeddings
    outputs = model(**inputs)

    return outputs.text_embeds

def similarity_search(query_embed, target_embeddings, content_list, k=5, threshold=0.05, temperature=0.5):
    """
    Perform similarity search over embeddings and return top k results.
    
    Args:
        query_embed (torch.Tensor): Query embedding
        target_embeddings (torch.Tensor): Target embeddings matrix to search over
        content_list (list): List of content items corresponding to embeddings
        k (int, optional): Number of top results to return. Defaults to 5.
        threshold (float, optional): Minimum similarity score threshold. Defaults to 0.1.
        temperature (float, optional): Temperature for softmax scaling. Defaults to 0.5.
    
    Returns:
        tuple: (results, scores) where:
            - results: List of top k content matches
            - scores: Corresponding similarity scores
    """
    # Compute the dot product similarity between the query and target embeddings
    similarities = torch.matmul(query_embed, target_embeddings.T)
    
    # Apply softmax normalization to the similarity scores using the specified temperature
    scores = torch.nn.functional.softmax(similarities/temperature, dim=1)
    
    # Sort the results by similarity score in descending order
    sorted_indices = scores.argsort(descending=True)[0]
    sorted_scores = scores[0][sorted_indices]
    
    # Filter results below the threshold and select the top k matches
    filtered_indices = [
        idx.item() for idx, score in zip(sorted_indices, sorted_scores) 
        if score.item() >= threshold
    ][:k]
    
    # Retrieve the actual content items corresponding to the top indices
    top_results = [content_list[i] for i in filtered_indices]
    result_scores = [scores[0][i].item() for i in filtered_indices]
    
    return top_results, result_scores

def construct_prompt(query, text_results, image_results):
    """
    Construct a prompt for the LLM to generate a response, optimized for
    repetition avoidance and concise summarization.
    """
    # 1. Build a refined text context
    text_context = ""
    if text_results:
        text_context = "## DOCUMENT TEXT (Primary Context):\n\n"
        
        for i, t in enumerate(text_results[:12]):
            snippet = t.get('text', '').strip()
            title = t.get('snippet_title')
            section = t.get('section', '')
            
            # --- Anti-Repetition/Cleaning Step ---
            # Aggressively remove list continuations from the source
            snippet = snippet.replace('., etc.', '.').replace(', etc.', '.').replace('...', '.')
            
            # Stricter Truncation
            MAX_LEN = 250 # Reduced from 300 for increased safety
            if len(snippet) > MAX_LEN:
                # Truncate and ensure it ends cleanly with a period/marker
                snippet = snippet[:MAX_LEN].rsplit(' ', 1)[0] + "."
            
            # Use strong separation (double newline) to prevent list merging
            text_context += f"--- TEXT SNIPPET {i+1} ---\n"
            text_context += f"Source: {title} | {section}\n"
            text_context += f"Content: {snippet}\n\n"
            
    # 2. Summarize image captions only
    image_context = ""
    if image_results:
        image_context = "## IMAGE CAPTIONS (Secondary Context):\n\n"
        for i, im in enumerate(image_results[:3]):
            caption = im.get('caption')
            title = im.get('snippet_title')
            section = im.get('section')
            # Use strong separation here too
            image_context += f"--- IMAGE SNIPPET {i+1} ---\n"
            image_context += f"Source: {title} | {section}\n"
            image_context += f"Caption: {caption}\n\n"

    # 3. Final prompt with explicit anti-repetition instruction
    return f"""You are a concise assistant whose job is to answer the user's question using ONLY the document's extracted text.

USER QUERY: "{query}"

---

DOCUMENT TEXT (Primary Context):
{text_context if text_context else 'No primary text available.'}

---

IMAGE INFO (Secondary Context - captions only):
{image_context if image_context else 'No image captions available.'}

---

INSTRUCTIONS (MUST follow exactly):
1) Use ONLY the **DOCUMENT TEXT** above to answer the user's query. Do NOT use or describe images unless the user explicitly asks for image analysis.
2) Provide a single, direct answer to the user's question. Be concise (one or two short sentences) and do NOT add extra commentary.
3) **STRICTLY DO NOT REPEAT PHRASES OR LIST MARKERS (e.g., 'etc.', 'Development of', 'Data analysis and') multiple times in your final answer.**
4) If the document does not contain an answer, reply: "I couldn't find that information in the document." (nothing else).

ANSWER:"""


def context_retrieval(query, text_embeddings, image_embeddings, text_content_list, image_content_list, 
                    text_k=20, image_k=3, 
                    text_threshold=0.005, image_threshold=0.15,
                    text_temperature=0.35, image_temperature=0.6):
    """
    Perform context retrieval over embeddings and return top k results.
    
    Args:
        query (str): The user's query
        text_embeddings: Text embeddings to search over
        image_embeddings: Image embeddings to search over
        text_content_list: List of text content items
        image_content_list: List of image content items
        text_k (int): Number of top text results to retrieve (default: 20)
        image_k (int): Number of top image results to retrieve (default: 3)
        text_threshold (float): Minimum similarity threshold for text (default: 0.005)
        image_threshold (float): Minimum similarity threshold for images (default: 0.15)
        text_temperature (float): Temperature for text similarity softmax (default: 0.35)
        image_temperature (float): Temperature for image similarity softmax (default: 0.6)
    """
    # Generate an embedding for the user's query using the CLIP model
    query_embed = embed_text(query)

    # Perform similarity search for the query against both text and image embeddings
    text_results, _ = similarity_search(query_embed, text_embeddings, text_content_list, k=text_k, threshold=text_threshold, temperature=text_temperature)
    image_results, _ = similarity_search(query_embed, image_embeddings, image_content_list, k=image_k, threshold=image_threshold, temperature=image_temperature)

    return text_results, image_results
