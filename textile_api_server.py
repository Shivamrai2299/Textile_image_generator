from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import json
import shutil
import os
from dotenv import load_dotenv
import cloudinary
load_dotenv()
from Textile_Design_Generator import (
    create_prompt,
    generate_with_reference_gemini,
    generate_without_reference_gemini,
    edit_image_gemini,
    tile_image_grid,
    ColorIndex,
    edit_colors_from_url,
    replace_colors_on_indexed_image,
    edit_image_deep_ai,
    create_prompt_using_ref_image,
    generate_without_reference_deepai,
    get_eps,
    export_to_psd 
)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Load API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(" OPENAI_API_KEY not found in environment variables")


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("tmp_refs", exist_ok=True)


@app.post("/create_image")
async def create_image(
    description: str = Form(None),
    style: str = Form(None),
    color_info: str = Form(None),
    simplicity: int = Form(None),
    n_options: int = Form(None),
    reference_urls: str = Form(None)
):
    try:
        ref_paths = []

        if reference_urls:
            url_list = [url.strip() for url in reference_urls.split(",") if url.strip()]
            ref_paths.extend(url_list)

        if ref_paths:
            prompt = create_prompt_using_ref_image(description, style, color_info, simplicity)
            images = generate_with_reference_gemini(prompt, ref_paths, n_options)
        else:
            prompt = create_prompt(description, style, color_info, simplicity)
            images = generate_without_reference_deepai(prompt, n_options)

        if not images:
            raise ValueError("No images were generated. Please check your generator functions.")

        return {
            "message": f"{len(images)} image(s) generated",
            "prompt_used": prompt,
            "cloudinary_urls": images
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/edit_image")
async def edit_image(
    description: str = Form(...),
    image: str = Form(None)
):
    url = image.strip()

    edited = edit_image_deep_ai(description, url)
    if edited:
        return {"message": "Image edited successfully", "filename": f"{edited}"}
    else:
        return {"error": "Editing failed"}


@app.post("/create_index")
async def create_index(
    image: str = Form(None)
):
    url = image.strip()

    return edit_colors_from_url(url)

@app.post("/edit_colors")
async def edit_colors_endpoint(
    image: str = Form(...),  # Cloudinary URL of the indexed image
    palette_json_url: str = Form(...),  # Cloudinary URL of the uploaded palette JSON
    indices: str = Form(...),  # e.g., "0,2"
    hex_colors: str = Form(...)  # e.g., "#FF0000,#00FF00"
):
    try:
        indexed_image_url = image.strip()
        palette_url = palette_json_url.strip()

        index_list = list(map(int, indices.strip().split(',')))
        color_list = [c.strip() for c in hex_colors.strip().split(',')]

        if len(index_list) != len(color_list):
            return JSONResponse(
                status_code=400,
                content={"error": "Number of indices and hex colors must match."}
            )

        result = replace_colors_on_indexed_image(indexed_image_url, palette_url, index_list, color_list)
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tile_image_grid")
async def tile_image_api(
    image: str = Form(...),
    rows: int = Form(5),
    cols: int = Form(5)
):
    url = image.strip()

    tiled = tile_image_grid(url, grid_rows=rows, grid_cols=cols)

    if tiled:
        return {"message": "Image tiled", "filename": tiled}
    else:
        return {"error": "Tiling failed"}

@app.post("/convert_to_eps")
async def eps_creation(
    image: str = Form(...)
):
    url = image.strip()    
    cdr_image = get_eps(url)
    
    if cdr_image:
        return {"message": "EPS Image Generated", "filename": cdr_image}
    else:
        return {"error": "EPS Conversion failed"}

@app.post("/create_image_using_gemini")
async def gemini_image_creation(
    description: str = Form(None),
    style: str = Form(None),
    color_info: str = Form(None),
    simplicity: int = Form(None),
    n_options: int = Form(None),
    reference_urls: str = Form(None)
):
    
    ref_paths = []

    if reference_urls:
        url_list = [url.strip() for url in reference_urls.split(",") if url.strip()]
        ref_paths.extend(url_list)

    if ref_paths:
        prompt = create_prompt_using_ref_image(description, style, color_info, simplicity)
        print("prompt: " + prompt)
        images = generate_with_reference_gemini(prompt, ref_paths, n_options)
    else:
        prompt = create_prompt(description, style, color_info, simplicity)
        print("prompt: " + prompt)
        images = generate_without_reference_gemini(prompt, n_options)

    #filenames = [f"single_image_{i}.png" for i in range(len(images))]
    return {"message": f"{len(images)} image(s) generated", "cloudinary_urls": images}


@app.get("/")
def read_root():
    return {"message": "Textile Design Generator API is running."}
