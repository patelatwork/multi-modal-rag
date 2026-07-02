import base64
import mimetypes
from pathlib import Path

def encode_image(image_path: str | Path) -> str:
    """
    Encodes an image file to a base64 string.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        A base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_image_mime_type(image_path: str | Path) -> str:
    """
    Gets the MIME type of an image file based on its extension.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        MIME type string (e.g., 'image/png', 'image/jpeg').
    """
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/jpeg"

def load_image_base64_data_url(image_path: str | Path) -> str:
    """
    Loads an image and returns its base64 data URL representation.
    This format is suitable for passing to multimodal LLM APIs (e.g., OpenRouter, OpenAI, Fireworks).
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Data URL string (e.g., 'data:image/png;base64,...').
    """
    mime_type = get_image_mime_type(image_path)
    base64_str = encode_image(image_path)
    return f"data:{mime_type};base64,{base64_str}"
