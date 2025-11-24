import fitz
import os
import json
import re
from transformers import CLIPProcessor, CLIPTextModelWithProjection
import torch

#Split text into chunks with overlap.
def split_text_with_overlap(text, max_length=256, overlap_percentage=0.25):
    if len(text) <= max_length:
        return [text]
    
    overlap_size = int(max_length * overlap_percentage)
    chunks = []
    
    # Calculate the starting indices for each chunk based on max_length and overlap
    chunk_starts = range(0, len(text), max_length - overlap_size)
    
    for start in chunk_starts:
        chunk = text[start:start + max_length]
            
        # Attempt to break the chunk at the last space to avoid splitting words
        if start + max_length < len(text):
            last_space = chunk.rfind(' ')
            if last_space != -1:
                chunk = chunk[:last_space]
            
        chunks.append(chunk.strip())
    
    return chunks

def get_section_header(chunk):
    """
    Identifies a known resume section header at the start of a text chunk.
    If no major header is found, it finds the first capitalized line or defaults.
    """
    # Define major resume sections (case-insensitive search for flexibility)
    MAJOR_HEADERS = {
        "EDUCATION": "EDUCATION",
        "SKILLS": "SKILLS",
        "PROFESSIONAL EXPERIENCE": "PROFESSIONAL EXPERIENCE",
        "LANGUAGES": "LANGUAGES",
        "INTERESTS": "INTERESTS",
    }
    
    # 1. Check for a major header at the start of the chunk
    for pattern, title in MAJOR_HEADERS.items():
        if re.search(r'^\s*' + re.escape(pattern) + r'\s*$', chunk[:50].strip(), re.IGNORECASE):
            return title
            
    # 2. Check for a specific project/item title (e.g., "P2IP Project - ESILV")
    # This captures the first few non-header lines which often name the item
    first_line = chunk.split('\n')[0].strip()
    if len(first_line) > 5 and len(first_line) < 60 and first_line.isupper() and not first_line in MAJOR_HEADERS.values():
         return first_line
         
    # 3. Default for snippets at the start (usually contact/summary)
    if 'Engineering Student' in chunk[:100]:
        return "Summary & Contact"
        
    return None # Return None if no suitable header is found

def parse_pdf_content(pdf_path):
    # ... (existing setup code)
    doc = fitz.open(pdf_path)
    article_title_base = os.path.basename(pdf_path) # Use this as a fallback source ID
    structured_content = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_chunks = split_text_with_overlap(text)
            
            # Track the most recent major header found
            current_major_header = article_title_base
            
            for chunk in text_chunks:
                # 1. Identify the new, better title
                new_title = get_section_header(chunk)
                
                if new_title:
                    # If we found a major header, update the tracker
                    if new_title in ["EDUCATION", "SKILLS", "PROFESSIONAL EXPERIENCE", "LANGUAGES", "INTERESTS"]:
                        current_major_header = new_title
                    else:
                        # If it's a project title, use a combination for better context
                        current_major_header = f"{current_major_header}: {new_title}"

                # 2. Use the best available title for the snippet
                snippet_title_to_use = new_title if new_title else current_major_header
                
                structured_content.append({
                    # Key change: Use the section name as the snippet title
                    'snippet_title': snippet_title_to_use, 
                    'section': f"Page {page_num + 1}",
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

# Convet text to embeddings using CLIP
def embed_text(text):
    """
        
    """
    
    # Load the pre-trained CLIP text model for generating embeddings
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    # Load the CLIP processor for tokenizing text and preprocessing images
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
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
                snippet = snippet[:MAX_LEN].rsplit(' ', 1)[0] + "..."
            
            # Use strong separation (double newline) to prevent list merging
            text_context += f"--- TEXT SNIPPET {i+1} ---\n"
            text_context += f"Source: {title} | {section}\n"
            text_context += f"Content: {snippet}\n\n"
            
    # 2. Summarize image captions only
    image_context = ""
    if image_results:
        image_context = "## IMAGE CAPTIONS (Secondary Context):\n\n"
        for i, im in enumerate(image_results[:3]):
            caption = im.get('caption', 'No caption')
            title = im.get('snippet_title' '')
            section = im.get('section', '')
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
