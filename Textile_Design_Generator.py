import os
import base64  # Used to encode images into text for processing
import io  # Helps handle data streams like image bytes
from io import BytesIO  # Creates a memory buffer for image data
import numpy as np  # Library for numerical operations, like handling image pixel data
import matplotlib.pyplot as plt  # Used for creating visualizations like graphs and images
from PIL import Image, ImageFilter  # Library for opening and editing images
from sklearn.cluster import KMeans  # Machine learning tool to group similar colors
import webcolors  # Helps convert colors to names (e.g., RGB to "red")
from matplotlib.colors import rgb2hex  # Converts RGB colors to hex codes
import requests  # Used to download images from URLs
from PIL import Image
import numpy as np
from psd_tools.api.psd_image import PSDImage
from psd_tools.api.layers import Group, PixelLayer

from google import genai  # Google's library for generative AI models
#import google.generativeai as genai # Google's AI library for image generation
from google.genai import types  # Types for configuring Google's AI responses
import openai  # OpenAI's library for DALL·E image generation
import anthropic  # Anthropic's library for Claude AI model
import cloudinary
import cloudinary.uploader
import mimetypes
import tempfile
import json
import subprocess
import uuid

cloudinary.config(
    cloud_name="dt1lqqhk7",       # e.g., "mydesigncloud"
    api_key="992317552783136",
    api_secret="wc-paoCXdKuogK5p2FTKi_SKMqw",
    secure=True
)

BASE_DIR = "/home/testuserclaw/claw/Textile_Design_Generator/created_images/"

# Load API keys for AI services from environment variables or use default keys
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "AIzaSyBPbGIBe5t87Ye64Ag5xTj80BJtJpzoUXk"
)  # Key for Google's Gemini AI
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "sk-proj-gAApXoHKg8FdtbfpNKoqEhS89sQcsP3KGFzgFYi2MQK8AKWcfRfEsSBhnWkmBepl_"
    "pbO-RXAQnT3BlbkFJC4mOXL-QE5qrRnuUGcp26_W5CDwChqlGmvGhvqn6kueRhJusKY5fsnHU6micf"
    "9-_Ra8b_gNUsA"
)  # Key for OpenAI's DALL·E
CLAUDE_API_KEY = os.getenv(
    "CLAUDE_API_KEY",
    "sk-ant-api03-6BiCtFbezvg6swwhPJavFwjeQXx4vskEaX5WcJ05PbzUO7BUoR5zZ7npekIqqTkjbkg"
    "-RLlBV8L4xPXCEaGUlg-BnFlNAAA"
)  # Key for Anthropic's Claude

# Initialize clients for each AI service using their respective API keys
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
#genai.configure(api_key=GEMINI_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def export_to_psd(image_path, color_mapping, output_psd="output_layers.psd"):
        """
    Export indexed image to PSD with separate layers per color.
        Each unique color in color_mapping will become its own Photoshop layer.

        Args:
            image_path (str): Path to the indexed PNG/JPG.
            color_mapping (dict): {index: "#HEX"} mapping of colors.
            output_psd (str): Output PSD file path.
        """
        # Load image
        base_img = Image.open(image_path).convert("RGB")
        img_array = np.array(base_img)

        # Create empty PSD
        psd = PSDImage.new(mode="RGB", size=base_img.size)

        # Add one layer per color
        for idx, hex_color in color_mapping.items():
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

            # Create mask for this color
            mask = (img_array == [r, g, b]).all(axis=-1).astype(np.uint8) * 255
            if mask.sum() == 0:
                continue

            # Create new blank image with only this color
            color_layer = np.zeros_like(img_array, dtype=np.uint8)
            color_layer[mask == 255] = [r, g, b]
            pil_layer = Image.fromarray(color_layer, "RGB")

            # Add as a Photoshop layer
            layer = PixelLayer.from_PIL(pil_layer, name=f"Color_{idx}_{hex_color}")
            psd.append(layer)

        # Save PSD
        psd.save(output_psd)
        print(f"[INFO] PSD saved: {output_psd}")
        return output_psd


# Class to analyze and modify colors in an image
class ColorIndex:
    def __init__(self, image_path, k=10):
        # Initialize with image path and number of colors to extract (default 10)
        self.image_path = image_path
        self.k = k
        self.image = Image.open(image_path).convert("RGB")  # Open image in RGB format
        self.width, self.height = self.image.size  # Get image dimensions
        self.image_array = np.array(self.image)  # Convert image to a numerical array
        self.flattened_pixels = self.image_array.reshape(-1, 3)  # Flatten pixels for clustering
        self.index_map = None  # Will store the color index map
        self.cluster_colors = None  # Will store the main colors found
        self.color_distribution = None  # Will store percentage of each color
        self.color_names = {}  # Will store color details (RGB, hex, name)
        self.label_to_index = {}  # Will map labels to color indices
        self.generate_index_map()  # Run color analysis

    def generate_index_map(self):
        # Analyze image to find the main colors
        print(f"[INFO] Generating index map with {self.k} colors...")
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)  # Use KMeans to group pixels into k colors
        kmeans.fit(self.flattened_pixels)  # Cluster pixels by color
        self.cluster_colors = kmeans.cluster_centers_.astype(int)  # Get the main colors
        labels = kmeans.labels_  # Get which cluster each pixel belongs to
        self.index_map = labels.reshape(self.height, self.width)  # Reshape labels to match image size

        # Calculate the percentage of each color in the image
        unique, counts = np.unique(labels, return_counts=True)
        total_pixels = self.width * self.height
        self.color_distribution = {
            i: count/total_pixels*100 for i, count in zip(unique, counts)
        }

        # Sort colors by how common they are
        sorted_indices = sorted(
            self.color_distribution.keys(),
            key=lambda x: self.color_distribution[x],
            reverse=True
        )
        new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}

        # Reassign indices to prioritize dominant colors
        new_index_map = np.zeros_like(self.index_map)
        for old_idx, new_idx in new_mapping.items():
            new_index_map[self.index_map == old_idx] = new_idx
        self.index_map = new_index_map

        # Update cluster colors to match sorted order
        self.cluster_colors = np.array([
            self.cluster_colors[old_idx] for old_idx in sorted_indices
        ])
        self.color_distribution = {
            new_mapping[old_idx]: self.color_distribution[old_idx]
            for old_idx in self.color_distribution
        }

        # Assign names and hex codes to each color
        for idx, color in enumerate(self.cluster_colors):
            color_tuple = tuple(color)
            hex_code = rgb2hex(color/255)  # Convert RGB to hex
            color_name = self.get_color_name(color_tuple)  # Get human-readable color name
            self.color_names[idx] = {
                'rgb': color_tuple,
                'hex': hex_code,
                'name': color_name,
                'percentage': self.color_distribution[idx]
            }

    def get_color_name(self, rgb_tuple):
        # Try to find an exact color name, or the closest match
        try:
            return webcolors.rgb_to_name(rgb_tuple)
        except ValueError:
            min_distance = float('inf')
            closest_name = None
            for key, value in webcolors.CSS3_HEX_TO_NAMES.items():
                rgb = webcolors.hex_to_rgb(key)
                distance = sum((a - b) ** 2 for a, b in zip(rgb_tuple, rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_name = value
            return closest_name

    # def save_indexed_image(self, output_path="indexed_image.png"):
    #     Create a simplified image using only the main colors
    #    indexed_image = np.zeros_like(self.image_array)
    #    for i in range(self.k):
    #        mask = np.zeros((self.height, self.width, 3), dtype=bool)
    #        mask[:, :, 0] = mask[:, :, 1] = mask[:, :, 2] = (self.index_map == i)
    #        indexed_image[mask] = self.cluster_colors[i].reshape(-1)[0]

    #     Image.fromarray(indexed_image.astype('uint8')).save(output_path)
    #     print(f"[INFO] Saved indexed image to {output_path}")
    #     return output_path
    
    def save_indexed_image(self, output_path="indexed_image.png"):
    # Create an empty RGB image
        indexed_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Assign each cluster color to the right pixels
        for i in range(self.k):
            indexed_image[self.index_map == i] = self.cluster_colors[i]

        Image.fromarray(indexed_image).save(output_path)
        print(f"[INFO] Saved indexed image to {output_path}")
        return output_path

    def save_index_map(self, output_path="index_map.png", colormap='viridis'):
        # Save a visual map showing where each color is used
        plt.figure(figsize=(10, 10))
        plt.imshow(self.index_map, cmap=colormap)
        plt.colorbar(label='Color Index')
        plt.title('Color Index Map')
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] Saved index map visualization to {output_path}")
        return output_path


    def export_color_palette(self, output_path="color_palette.txt"):
        # Save a text file listing the main colors and their details
        with open(output_path, "w") as f:
            f.write("Index | Hex Code | Approx. Name | % of Image\n")
            f.write("-" * 50 + "\n")
            for idx in range(self.k):
                info = self.color_names[idx]
                f.write(
                    f"{idx} | {info['hex']} | {info['name']} | "
                    f"{info['percentage']:.2f}%\n"
                )
        print(f"[INFO] Exported color palette to {output_path}")
        return output_path

    def visualize_palette(self):
        # Show a bar chart of the color distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        indices = list(range(self.k))
        percentages = [self.color_names[i]['percentage'] for i in indices]
        colors = [tuple(c/255) for c in self.cluster_colors]

        bars = ax.bar(indices, percentages, color=colors)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            hex_code = self.color_names[i]['hex']
            percentage = self.color_names[i]['percentage']
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.5,
                f"{hex_code}\n{percentage:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_xlabel('Color Index')
        ax.set_ylabel('Percentage of Image (%)')
        ax.set_title('Color Distribution in Image')
        ax.set_xticks(indices)
        ax.set_xticklabels(indices)
        plt.tight_layout()
        plt.show()

    def assign_label(self, label, index):
        # Assign a custom label to a color index
        if 0 <= index < self.k:
            self.label_to_index[label.lower()] = index
            print(
                f"[INFO] Assigned label '{label}' to color index {index} "
                f"({self.color_names[index]['name']})"
            )
        else:
            print(f"[ERROR] Index {index} out of range (0-{self.k-1})")

    def get_index_by_label(self, label):
        # Get the color index for a given label
        return self.label_to_index.get(label.lower())

    def change_color(self, target_index, new_color, output_path=None):
        # Change a specific color in the image
        if isinstance(target_index, str):
            if target_index.lower() in self.label_to_index:
                target_index = self.label_to_index[target_index.lower()]
            else:
                print(f"[ERROR] Label '{target_index}' not found")
                return None

        if isinstance(new_color, str) and new_color.startswith('#'):
            new_color = webcolors.hex_to_rgb(new_color)
        new_color = np.array(new_color)

        modified_array = self.image_array.copy()
        mask = np.zeros((self.height, self.width, 3), dtype=bool)
        mask[:,:,0] = mask[:,:,1] = mask[:,:,2] = (self.index_map == target_index)

        for c in range(3):
            modified_array[:,:,c][self.index_map == target_index] = new_color[c]

        modified_image = Image.fromarray(modified_array.astype('uint8'))
        if output_path:
            modified_image.save(output_path)
            print(f"[INFO] Saved modified image to {output_path}")

        return modified_image

    def bulk_color_change(self, color_mapping, output_path=None):
        # Change multiple colors at once
        modified_array = self.image_array.copy()

        for target, new_color in color_mapping.items():
            if isinstance(target, str):
                if target.lower() in self.label_to_index:
                    target = self.label_to_index[target.lower()]
                else:
                    print(f"[WARNING] Label '{target}' not found, skipping")
                    continue

            if isinstance(new_color, str) and new_color.startswith('#'):
                new_color = webcolors.hex_to_rgb(new_color)
            new_color = np.array(new_color)

            for c in range(3):
                modified_array[:,:,c][self.index_map == target] = new_color[c]

        modified_image = Image.fromarray(modified_array.astype('uint8'))
        if output_path:
            modified_image.save(output_path)
            print(f"[INFO] Saved modified image to {output_path}")

        return modified_image

# Detect the type of image file (e.g., JPEG, PNG)
def detect_mime_type(image_path):
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        return "image/jpeg"    
    elif image_path.lower().endswith('.png'):
        return "image/png"
    elif image_path.lower().endswith('.webp'):
        return "image/webp"
    return "image/jpeg"


# Helper to encode image from Cloudinary URL
def encode_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        mime_type = response.headers.get('Content-Type', mimetypes.guess_type(image_url)[0])
        encoded_image = base64.b64encode(response.content).decode('utf-8')
        return encoded_image, mime_type
    except Exception as e:
        print(f"[ERROR] Failed to fetch or encode image from URL: {e}")
        return None, None

# Convert an image to base64 text for AI processing
def encode_image(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            mime_type = detect_mime_type(image_path)
            return encoded_image, mime_type
    except FileNotFoundError:
        print(f"[ERROR] Image file '{image_path}' not found.")
        return None, None
    except Exception as e:
        print(f"[ERROR] Error encoding image: {e}")
        return None, None

# Save an AI-generated image to a file
def save_generated_image(image_data, filename="generated_image.png"):
    try:
        if isinstance(image_data, bytes):
            decoded_data = image_data
        else:
            decoded_data = base64.b64decode(image_data)

        image = Image.open(BytesIO(decoded_data)).convert("RGB")
        image.save(BASE_DIR + filename, dpi=(300, 300))
        print(f"[INFO] Image saved as: {filename}")
        return BASE_DIR + filename
    except Exception as e:
        print(f"[ERROR] Error processing generated image: {e}")
        return None

# Upload Generated Image
def upload_to_cloudinary(local_path):
    try:
        result = cloudinary.uploader.upload(local_path)
        print(result)
        print("Image uploaded to Cloudinary")
        return result.get("secure_url")
    except Exception as e:
        print(f"[ERROR] Failed to upload to Cloudinary: {e}")
        return None

def upload_json_to_cloudinary(json_path):
    response = cloudinary.uploader.upload(
            json_path,
            resource_type='raw',
            overwrite=True
        )

    secure_url = response['secure_url']
    print(f"[SUCCESS] JSON uploaded: {secure_url}")
    return secure_url

def get_eps(image_url):
    try:
        # Step 0: Generate a unique output name
        uid = str(uuid.uuid4())[:8]
        eps_path = f"temp_{uid}.eps"

        # Step 1: Download image from URL
        print(f"[INFO] ({uid}) Downloading image...")
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Step 2: Save as EPS locally
        img.save(eps_path, format='EPS')

        # Step 3: Upload EPS to Cloudinary
        print(f"[INFO] ({uid}) Uploading EPS to Cloudinary...")
        eps_url = upload_to_cloudinary(eps_path)

        # Step 4: Clean up
        if os.path.exists(eps_path):
            os.remove(eps_path)

        return eps_url

    except Exception as e:
        print(f"[ERROR] Failed to convert image to EPS: {e}")
        return None

# Create a prompt for generating textile designs
def create_prompt(description: str, style: str, color_info: str, simplicity: int):
    prompt = f"""
    You are an expert in generating **high-quality, Pinterest-level image prompts** for textile surface design, suitable for premium fashion, home decor, and artistic showcases.
    
     Requirements:
    - **Description:** {description}
    - **Style:** {style} (standalone or repetitive pattern)
    - **Colors:** {color_info} (specific number or range)
    - **Simplicity Level:** {simplicity}/10 (1=extremely simple, 10=highly intricate)

     **Quality Guidelines**:
    - The design must be of **Pinterest-worthy quality**: visually stunning, artistically polished, with harmonious composition and refined color palette.
    - The output should be **realistic, professional, and ready for premium textile production or digital showcasing**.
    - No basic or generic elements; the design should feel **inspired and curated**, like it belongs in a designer catalog or an editorial spread.

     **Repetitive pattern rules**:
    - The pattern must be perfectly seamless and tileable across all edges — invisible joins, no mirroring effects, no visible grid repetitions.
    - Natural flow: elements on edges should wrap organically, not look copied or mirrored.
    - The design should **avoid artificial symmetry** and instead use artistic variation to create visual interest.

     **Standalone rules**:
    - The motif should be **balanced and centered with generous margins** — no cropped or cut-off parts.
    - The standalone element should have an elegant layout with purposeful negative space, like a premium print or logo.

    🖌 **Simplicity control**:
    - 1–3: Minimalist with elegant whitespace, 1–2 clean elements, solid colors, luxurious simplicity.
    - 4–6: Lightly decorated, tasteful accents, balanced density.
    - 7–10: Richly layered, intricate details, realistic textures, artistic shadows, depth.

     **Color rules**:
    - Use {color_info} distinct hex colors.
    - Ensure colors are harmonious, rich, and aesthetically pleasing.

     Output your result in this **strict structured format**:

    ---
    Description: [One-sentence overview of the image design]
    Style: {style} (either 'standalone' or 'repetitive pattern')
    Colors: {color_info} (expressed as a list of 3–7 distinct hex colors, e.g. #FF6B6B, #27AE60, etc.)
    Complexity Level: {simplicity}/10

    Design Instructions:
    - [Bullet point 1]
    - [Bullet point 2]
    - [Bullet point 3]
    - [Bullet point 4]
    ...
    ---

    Now generate a detailed, **Pinterest-level textile design prompt** that meets these criteria:
"""

    # Build a detailed instruction set for the AI to create a textile design
    # prompt = f"""
    # You are an expert in generating image prompts for textile surface design.
    # Requirements:
    # - Description: {description}
    # - Style: {style} (standalone or repetitive pattern)
    # - Colors: {color_info} (specific number or range)
    # - Simplicity Level: {simplicity}/10 (1=extremely simple, 10=highly intricate)

    # For 'repetitive pattern':
    # - Must be perfectly seamless and tileable across all edges.
    # - Elements touching left edge continue on right edge; top edge continues on bottom.
    # - Avoid centered motifs unless they extend beyond edges and wrap seamlessly.
    # - Prevent visible seams; avoid mirroring or grid-aligned clones.
    # - Slightly vary element positions to avoid uniform repetition.

    # For 'standalone':
    # - Center motif with ample margin, not cropped by edges.
    # - Design should be balanced within the frame, non-repeating.

    # Simplicity control:
    # - 1–3: 1–2 elements, 40–50% whitespace, no clutter, solid-filled lines.
    # - 4–6: 2–3 elements, light decorations, balanced with breathing room.
    # - 7–10: Full intricacy with overlapping elements, texture, shadows.

    # Use only solid fills unless gradients specified. Adhere to {color_info} distinct hex colors.
    
    # Your task is to output the prompt in a **strictly structured format** like below:

    # ---
    # Description: [One-sentence overview of the image design]
    # Style: {style} (either 'standalone' or 'repetitive pattern')
    # Colors: {color_info} (expressed as a list of 3–7 distinct hex colors, e.g. #FF6B6B, #27AE60, etc.)
    # Complexity Level: {simplicity}/10
    
    # Design Instructions:
    # - [Bullet point 1]
    # - [Bullet point 2]
    # - [Bullet point 3]
    # - [Bullet point 4]
    # ...
    # ---
    
    # Now Generate the textile design prompt, from here:
    
    # """

    # Use Claude AI to refine the prompt
    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.content[0].text

# Create Image descriptions
def describe_images(image_urls):
    descriptions = []

    for url in image_urls:
        try:
            print(f"[INFO] Fetching and describing: {url}")
            response = requests.get(url)
            if response.status_code != 200:
                print(f"[WARNING] Skipped {url} - fetch failed")
                continue
            img_bytes = response.content
            b64_img = base64.b64encode(img_bytes).decode("utf-8")

            # Use OpenAI GPT-4 Vision
            vision_resp = openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this textile image in one short sentence relevant to its design."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }
                ],
                max_tokens=150
            )

            vision_text = vision_resp.choices[0].message.content.strip()
            descriptions.append(vision_text)

        except Exception as e:
            print(f"[ERROR] Failed to describe image {url}: {e}")

    return descriptions

def create_prompt_using_ref_image(description: str, style: str, color_info: str, simplicity: int, image_urls: list = []):
    # Describe reference images
    image_descriptions = describe_images(image_urls) if image_urls else []

    # Combine user description with image insights
    enriched_description = description
    if image_descriptions:
        enriched_description += "\n\nInspiration from reference images:\n"
        for i, desc in enumerate(image_descriptions, 1):
            enriched_description += f"- Ref {i}: {desc}\n"

    # Main prompt to Claude

    full_prompt = f"""
    You are an expert in generating **high-quality, Pinterest-level image prompts** for textile surface design, suitable for premium fashion, home decor, and artistic showcases.
    
     Requirements:
    - **Description:** {description}
    - **Style:** {style} (standalone or repetitive pattern)
    - **Colors:** {color_info} (specific number or range)
    - **Simplicity Level:** {simplicity}/10 (1=extremely simple, 10=highly intricate)

     **Quality Guidelines**:
    - The design must be of **Pinterest-worthy quality**: visually stunning, artistically polished, with harmonious composition and refined color palette.
    - The output should be **realistic, professional, and ready for premium textile production or digital showcasing**.
    - No basic or generic elements; the design should feel **inspired and curated**, like it belongs in a designer catalog or an editorial spread.

     **Repetitive pattern rules**:
    - The pattern must be perfectly seamless and tileable across all edges — invisible joins, no mirroring effects, no visible grid repetitions.
    - Natural flow: elements on edges should wrap organically, not look copied or mirrored.
    - The design should **avoid artificial symmetry** and instead use artistic variation to create visual interest.

     **Standalone rules**:
    - The motif should be **balanced and centered with generous margins** — no cropped or cut-off parts.
    - The standalone element should have an elegant layout with purposeful negative space, like a premium print or logo.

    🖌 **Simplicity control**:
    - 1–3: Minimalist with elegant whitespace, 1–2 clean elements, solid colors, luxurious simplicity.
    - 4–6: Lightly decorated, tasteful accents, balanced density.
    - 7–10: Richly layered, intricate details, realistic textures, artistic shadows, depth.

     **Color rules**:
    - Use {color_info} distinct hex colors.
    - Ensure colors are harmonious, rich, and aesthetically pleasing.

     Output your result in this **strict structured format**:

    ---
    Description: [One-sentence overview of the image design]
    Style: {style} (either 'standalone' or 'repetitive pattern')
    Colors: {color_info} (expressed as a list of 3–7 distinct hex colors, e.g. #FF6B6B, #27AE60, etc.)
    Complexity Level: {simplicity}/10

    Design Instructions:
    - [Bullet point 1]
    - [Bullet point 2]
    - [Bullet point 3]
    - [Bullet point 4]
    ...
    ---

    Now generate a detailed, **Pinterest-level textile design prompt** that meets these criteria:
"""

    # full_prompt = f"""
    # You are an expert in generating image prompts for textile surface design.
    # Requirements:
    # - Description: {enriched_description}
    # - Style: {style} (standalone or repetitive pattern)
    # - Colors: {color_info} (specific number or range)
    # - Simplicity Level: {simplicity}/10 (1=extremely simple, 10=highly intricate)

    # For 'repetitive pattern':
    # - Must be perfectly seamless and tileable across all edges.
    # - Elements touching left edge continue on right edge; top edge continues on bottom.
    # - Avoid centered motifs unless they extend beyond edges and wrap seamlessly.
    # - Prevent visible seams; avoid mirroring or grid-aligned clones.
    # - Slightly vary element positions to avoid uniform repetition.

    # For 'standalone':
    # - Center motif with ample margin, not cropped by edges.
    # - Design should be balanced within the frame, non-repeating.

    # Simplicity control:
    # - 1–3: 1–2 elements, 40–50% whitespace, no clutter, solid-filled lines.
    # - 4–6: 2–3 elements, light decorations, balanced with breathing room.
    # - 7–10: Full intricacy with overlapping elements, texture, shadows.

    # Use only solid fills unless gradients specified. Adhere to {color_info} distinct hex colors.
    
    # Your task is to output the prompt in a **strictly structured format** like below:

    # ---
    # Description: [One-sentence overview of the image design]
    # Style: {style} (either 'standalone' or 'repetitive pattern')
    # Colors: {color_info} (expressed as a list of 3–7 distinct hex colors, e.g. #FF6B6B, #27AE60, etc.)
    # Complexity Level: {simplicity}/10
    
    # Design Instructions:
    # - [Bullet point 1]
    # - [Bullet point 2]
    # - ...
    # ---

    # Now Generate the textile design prompt, from here:
    # """

    # Call Claude Haiku
    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.3
    )

    return response.content[0].text

# Utility to safely delete a file if it exists
def delete_file_safely(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Deleted temp file: {path}")
    except Exception as e:
        print(f"[WARNING] Failed to delete temp file {path}: {e}")


# Generate images using Gemini AI with reference images SHIVAM RAI
def generate_with_reference_gemini(prompt, image_paths, n_options=1):
    images = []
    try:
        print("[INFO] Generating image(s) with Gemini and reference...")

        for _ in range(n_options):
            content_parts = [{"text": prompt}]

            for img_path in image_paths:
                if img_path.startswith("http://") or img_path.startswith("https://"):
                    response = requests.get(img_path)
                    if response.status_code == 200:
                        mime_type = response.headers.get("Content-Type") or "image/png"
                        encoded_image = base64.b64encode(response.content).decode("utf-8")
                    else:
                        print(f"[WARNING] Failed to fetch {img_path}, skipping.")
                        continue
                else:
                    encoded_image, mime_type = encode_image(img_path)
                    if not encoded_image:
                        continue

                content_parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded_image
                    }
                })

            # Call Gemini API
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=content_parts,
                response_modalities=["IMAGE"]  # ✅ correct way
            )

            print("[DEBUG] Gemini raw response:", response)

            if not response.candidates:
                print("[ERROR] Gemini returned no candidates")
                continue

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                print("[ERROR] Gemini candidate has no content parts")
                continue

            found_image = False
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    img_path = save_generated_image(
                        part.inline_data.data,
                        f"gemini_ref_{len(images)}.png"
                    )
                    if img_path:
                        url = upload_to_cloudinary(img_path)
                        images.append(url)
                        delete_file_safely(img_path)
                        found_image = True

            if not found_image:
                print("[WARNING] Gemini response contained no image inline_data")

    except Exception as e:
        print(f"[ERROR] Gemini image generation with reference failed: {e}")

    return images


# def generate_with_reference_gemini(prompt, image_paths, n_options=1):
#     images = []
#     try:
#         print("[INFO] Generating image(s) with Gemini and reference...")
#         for _ in range(n_options):
#             content_parts = [{"text": prompt}]

#             for img_path in image_paths:
#                 if img_path.startswith("http://") or img_path.startswith("https://"):
#                     # Download from Cloudinary (or any URL)
#                     response = requests.get(img_path)
#                     if response.status_code == 200:
#                         mime_type = response.headers.get("Content-Type", "image/jpeg")
#                         encoded_image = base64.b64encode(response.content).decode("utf-8")
#                     else:
#                         print(f"[WARNING] Failed to fetch {img_path}, skipping.")
#                         continue
#                 else:
#                     # Local image
#                     encoded_image, mime_type = encode_image(img_path)
#                     if not encoded_image:
#                         continue

#                 content_parts.append({
#                     "inline_data": {
#                         "mime_type": mime_type,
#                         "data": encoded_image
#                     }
#                 })

#             response = gemini_client.models.generate_content(
#                 model="gemini-2.0-flash-preview-image-generation",
#                 contents=content_parts,
#                 config=types.GenerateContentConfig(
#                     response_modalities=['TEXT', 'IMAGE']
#                 )
#             )

#             for part in response.candidates[0].content.parts:
#                 if part.inline_data is not None:
#                     img_path = save_generated_image(
#                         part.inline_data.data,
#                         f"gemini_ref_{len(images)}.png"
#                     )
#                     if img_path:
#                         url = upload_to_cloudinary(img_path)
#                         images.append(url)
#                         delete_file_safely(img_path)

#     except Exception as e:
#         print(f"[ERROR] Gemini image generation with reference failed: {e}")

#     return images


# Generate images using Gemini AI without reference images
def generate_without_reference_gemini(prompt, n_options=1):
    images = []
    try:
        print("[INFO] Generating image(s) with Gemini...")
        for _ in range(n_options):
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[{"text": prompt}],
                response_modalities=["IMAGE"]   # ✅ correct way
            )

            if not response.candidates:
                print("[ERROR] Gemini returned no candidates")
                continue

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                print("[ERROR] Gemini candidate has no content parts")
                continue

            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    img_path = save_generated_image(
                        part.inline_data.data,
                        f"gemini_{len(images)}.png"
                    )
                    if img_path:
                        url = upload_to_cloudinary(img_path)
                        images.append(url)
                        delete_file_safely(img_path)

    except Exception as e:
        print(f"[ERROR] Gemini image generation failed: {e}")

    return images

# Generate images using Deep AI without reference images SHIVAM RAI
def generate_without_reference_deepai(prompt, n_options=1):
    images = []
    os.makedirs("created_images", exist_ok=True)
    try:
        print("[INFO] Generating image(s) with Deep AI...")
        for i in range(n_options):
            r = requests.post(
                "https://api.deepai.org/api/text2img",
                data={
                    'text': prompt,
                    'image_generator_version': 'genius',
                    'genius_preference': 'graphic',
                    'width': '1024',
                    'height': '1024'
                },
                headers={'api-key': 'e380ba0d-18e2-432a-9aba-9878932f87ba'}
            )

            resp_json = r.json()
            output_url = resp_json.get('output_url') or resp_json.get('share_url')

            if output_url:
                # Download image
                img_data = requests.get(output_url).content
                local_path = f"created_images/deepai_image_{i}.png"
                with open(local_path, "wb") as f:
                    f.write(img_data)
                print(f"[INFO] Saved locally at {local_path}")

                # Upload to Cloudinary
                cloud_url = upload_to_cloudinary(local_path)
                if cloud_url:
                    images.append(cloud_url)
                    print(f"[INFO] Uploaded to Cloudinary: {cloud_url}")
                else:
                    images.append(output_url)  # fallback if upload fails
            else:
                print(f"[WARNING] No output_url in response: {resp_json}")

    except Exception as e:
        print(f"[ERROR] DeepAI image generation failed: {e}")

    return images
# def generate_without_reference_deepai(prompt, n_options=1):
#     images = []
#     try:
#         print("[INFO] Generating image(s) with Deep AI...")
#         for _ in range(n_options):
#             r = requests.post(
#                 "https://api.deepai.org/api/text2img",
#                 data={
#                     'text': prompt,
#                     'image_generator_version': 'genius',
#                     'genius_preference': 'graphic',
#                     'width': '1024',
#                     'height': '1024'
#                 },
#                 headers={'api-key': 'e380ba0d-18e2-432a-9aba-9878932f87ba'}
#             )
#             images.append(r.json().get('output_url') or r.json().get('share_url'))

#     except Exception as e:
#         print(f"[ERROR] DeepAI image generation failed: {e}")

#     return images

# Edit Image with Deep AI
def edit_image_deep_ai(description, image_url):
    r = requests.post(
        "https://api.deepai.org/api/image-editor",
        data={
            'image': image_url,
            'text': description,
        },
        headers={'api-key': 'e380ba0d-18e2-432a-9aba-9878932f87ba'}
    )
    url = r.json()['share_url']
    return url

# Generate images using DALL·E with reference images
def generate_with_reference_dalle(prompt, image_paths, n_options=1):
    """Generate images using DALL·E with reference images."""
    images = []
    try:
        print("[INFO] Generating image(s) with Gemini and reference...")
        for _ in range(n_options):
            content_parts = [{"text": prompt}]

            for img_path in image_paths:
                encoded_image, mime_type = encode_image(img_path)
                if encoded_image:
                    content_parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_image
                        }
                    })

            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )

            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    img = save_generated_image(
                        part.inline_data.data,
                        f"gemini_ref_{len(images)}.png"
                    )
                    if img:
                        images.append(img)

    except Exception as e:
        print(f"[ERROR] Gemini image generation with reference failed: {e}")

    return images


# Generate images using DALL·E without reference images
def generate_without_reference_dalle(prompt, n_options=1):
    images = []
    try:
        print("[INFO] Generating image(s) with DALL·E...")
        for i in range(n_options):
            try:
                response = openai_client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1
                )
            except Exception as img_error:
                print(f"[WARNING] gpt-image-1 failed, falling back to dall-e-3: {img_error}")

                response = openai_client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    n=1
                )

            if response.data:
                image_url = response.data[0].url
                image_response = requests.get(image_url)
                img = Image.open(BytesIO(image_response.content)).convert("RGB")
                images.append(img)
                img.save(f"dalle_{i}.png", dpi=(300, 300))
                print(f"[INFO] Generated image {i+1}/{n_options}")

    except Exception as e:
        print(f"[ERROR] DALL·E image generation failed: {e}")

    return images

# Edit an image using Gemini AI (Cloudinary URL supported)
    """
    Convert a color-indexed or edited image into an AutoCAD-editable DXF file.
    Each region in the image is converted into a polyline.
    """

    try:
        print("[INFO] Loading image for DXF conversion...")
        img = np.array(Image.open(image_path).convert("RGB"))

        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Threshold to isolate shapes
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours (each shape = polyline)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create new DXF document
        doc = ezdxf.new()
        msp = doc.modelspace()

        for cnt in contours:
            if len(cnt) >= 3:  # only valid polygons
                points = [(int(x), int(y)) for [x, y] in cnt.squeeze()]
                msp.add_lwpolyline(points, close=True)

        # Save DXF
        doc.saveas(dxf_output)
        print(f"[INFO] DXF file saved: {dxf_output}")

        return dxf_output

    except Exception as e:
        print(f"[ERROR] DXF conversion failed: {e}")
        return None
def edit_image_gemini(description, image_url):
    encoded_image, mime_type = encode_image_from_url(image_url)
    if not encoded_image:
        return None

    try:
        print("[INFO] Editing image with Gemini...")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[
                {"text": f"Edit this image: {description}"},
                {"inline_data": {"mime_type": mime_type, "data": encoded_image}}
            ],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                filename = f"edited_gemini_{os.path.basename(image_url.split('?')[0])}"
                img_path = save_generated_image(part.inline_data.data, filename)
                
                if img_path:
                    url = upload_to_cloudinary(img_path)
                    delete_file_safely(img_path)
                    return url

        print("[WARNING] No image generated in Gemini response.")
        return None

    except Exception as e:
        print(f"[ERROR] Gemini image editing failed: {e}")
        return None

# Edit an image using DALL·E
def edit_image_dalle(description, image_path):
    encoded_image, mime_type = encode_image(image_path)
    if not encoded_image:
        return None

    try:
        print("[INFO] Editing image with Gemini...")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[
                {"text": f"Edit this image: {description}"},
                {"inline_data": {"mime_type": mime_type, "data": encoded_image}}
            ],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return save_generated_image(
                    part.inline_data.data,
                    f"edited_gemini_{os.path.basename(image_path)}"
                )

        print("[WARNING] No image generated in Gemini response.")
        return None

    except Exception as e:
        print(f"[ERROR] Gemini image editing failed: {e}")
        return None

# Create a transparent mask for DALL·E editing
def create_transparent_mask(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size

        mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        mask_path = f"mask_{os.path.basename(image_path)}"
        mask.save(mask_path, format="PNG")
        return mask_path

    except Exception as e:
        print(f"[ERROR] Failed to create mask: {e}")
        raise

# Create a tiled image by repeating the input image in a grid
def tile_image_grid(image_url, grid_rows=5, grid_cols=5):
    try:
        # Step 1: Download the image from Cloudinary
        print("[INFO] Downloading tile image...")
        response = requests.get(image_url)
        response.raise_for_status()

        tmp_tile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_tile.write(response.content)
        tmp_tile.close()
        tile_path = tmp_tile.name

        # Step 2: Open image and get size
        tile = Image.open(tile_path)
        w, h = tile.size
        print(f"[INFO] Loaded tile image ({w}x{h})")

        # Step 3: Create blank result image
        result = Image.new("RGB", (w * grid_cols, h * grid_rows))
        for row in range(grid_rows):
            for col in range(grid_cols):
                result.paste(tile, (col * w, row * h))

        # Step 4: Save the tiled image temporarily
        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        output_path = output_tmp.name
        result.save(output_path)
        print(f"[INFO] Tiled image saved to temp path: {output_path}")

        # Step 5: Upload to Cloudinary
        cloud_url = upload_to_cloudinary(output_path)
        delete_file_safely(output_path)
        print(f"[SUCCESS] Tiled image uploaded to: {cloud_url}")

        return cloud_url

    except Exception as e:
        print(f"[ERROR] Failed to tile image: {e}")
        import traceback
        traceback.print_exc()
        return None

# Let the user assign labels to color indices
def assign_color_labels(indexer):
    print("\nAssign labels to color indices (optional):")
    print("Example: 'background 0' assigns the label 'background' to index 0")
    print("Enter label and index: foreground 1")
    print("Enter label and index: highlight 2")
    print("Enter label and index: background 0")
    print("Enter 'done' when finished.\n")

    while True:
        input_str = input("Enter label and index (or 'done'): ")
        if input_str.lower() == 'done':
            break

        try:
            parts = input_str.rsplit(' ', 1)
            if len(parts) != 2:
                print("[ERROR] Invalid format. Use 'label index' format.")
                continue

            label, index = parts[0], int(parts[1])
            indexer.assign_label(label, index)

        except ValueError:
            print("[ERROR] Index must be a number.")

# Edit colours from Cloudinary URL
def edit_colors_from_url(image_url, n_colors=10):
    try:
        print("[INFO] Downloading image from Cloudinary...")
        response = requests.get(image_url)
        response.raise_for_status()

        # Save image to a temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_file.write(response.content)
        tmp_file.close()
        image_path = tmp_file.name

        print(f"[INFO] Extracting {n_colors} colors from image...")
        indexer = ColorIndex(image_path, k=n_colors)

        original_indexed_image_path = indexer.save_indexed_image()
        indexed_image_path = upload_to_cloudinary(original_indexed_image_path)
        delete_file_safely(original_indexed_image_path)
        palette_file = indexer.export_color_palette()

        print("[INFO] Color palette and indexed image created.")

        # Build JSON response with color palette
        palette_json = []
        for idx in range(indexer.k):
            info = indexer.color_names[idx]
            palette_json.append({
                "index": idx,
                "hex": info['hex'],
                "name": info['name'],
                "percentage": round(info['percentage'], 2)
            })
        
        # Save to a temporary .json file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            json.dump(palette_json, tmp, indent=2)
            json_tmp_path = tmp.name
            
        pellet_url = upload_json_to_cloudinary(json_tmp_path)
        delete_file_safely(json_tmp_path)
        return {
            "indexed_image_path": indexed_image_path,
            "palette": pellet_url
        }

    except Exception as e:
        print(f"[ERROR] Color editing failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def replace_colors_on_indexed_image(indexed_image_url, palette_json_url, index_list, target_colors):
    try:
        # Step 1: Download indexed image
        print("[INFO] Downloading indexed image...")
        image_response = requests.get(indexed_image_url)
        image_response.raise_for_status()
        tmp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_image_file.write(image_response.content)
        tmp_image_file.close()
        image_path = tmp_image_file.name

        # Step 2: Download and parse original palette JSON
        print("[INFO] Downloading palette JSON...")
        json_response = requests.get(palette_json_url)
        json_response.raise_for_status()
        original_palette = json_response.json()

        # Step 3: Prepare dummy ColorIndex
        dummy = ColorIndex(image_path, k=len(original_palette))
        dummy.k = len(original_palette)

        # Step 4: Build full color_names dict and new_palette_json
        updated_palette = []
        index_to_color = dict(zip(index_list, target_colors))

        for item in original_palette:
            idx = item['index']
            new_hex = index_to_color.get(idx, item['hex'])  # Use replacement if provided, else original
            rgb = tuple(int(new_hex[i:i+2], 16) for i in (1, 3, 5))

            dummy.color_names[idx] = {
                'hex': new_hex,
                'name': item['name'],
                'percentage': item['percentage'],
                'rgb': rgb
            }

            updated_palette.append({
                'index': idx,
                'hex': new_hex,
                'name': item['name'],
                'percentage': item['percentage']
            })

        # Step 5: Apply replacements
        color_mapping = {
            item['index']: dummy.color_names[item['index']]['hex']
            for item in original_palette
        }
        output_filename = f"color_replaced_{os.path.basename(image_path)}"
        dummy.bulk_color_change(color_mapping, output_filename)

        # Step 6: Upload edited image
        edited_cloud_url = upload_to_cloudinary(output_filename)
        delete_file_safely(output_filename)

        # Step 7: Save updated_palette as temp .json and upload it
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_json:
            json.dump(updated_palette, tmp_json, indent=2)
            tmp_json_path = tmp_json.name

        updated_palette_url = upload_json_to_cloudinary(tmp_json_path)
        delete_file_safely(tmp_json_path)

        print(f"[INFO] Edited image: {edited_cloud_url}")
        print(f"[INFO] Updated palette: {updated_palette_url}")

        return {
            "edited_image_path": edited_cloud_url,
            "updated_palette_json_url": updated_palette_url,
            "replaced_colors": color_mapping
        }

    except Exception as e:
        print(f"[ERROR] Failed to apply color replacements: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Edit the colors in an image
def edit_colors(image_path):
    try:
        print("[INFO] Analyzing image colors...")
        try:
            n_colors = int(input("Enter number of colors to extract (default 10): ") or "10")
        except ValueError:
            print("[WARNING] Invalid number. Using default value of 10.")
            n_colors = 10

        indexer = ColorIndex(image_path, k=n_colors)

        indexed_image_path = indexer.save_indexed_image()  # Save the simplified image
        palette_file = indexer.export_color_palette()      # Save the color palette to a text file

        indexer.visualize_palette()  # Show a bar chart of the colors

        print(f"\n[INFO] Color palette saved to {palette_file}")
        print("[INFO] Please review the palette file to select color indices to modify.")

        print("\nColor Palette:")
        print("Index | Hex Code | Approx. Name | % of Image")
        print("-" * 50)
        for idx in range(indexer.k):
            info = indexer.color_names[idx]
            print(f"{idx} | {info['hex']} | {info['name']} | {info['percentage']:.2f}%")

        label_choice = input("\nDo you want to assign labels to color indices? (yes/no): ").lower()
        if label_choice == 'yes':
            assign_color_labels(indexer)

        index_str = input("\nEnter list of color indices to replace (e.g., 1,3,5): ")
        try:
            index_list = [int(i.strip()) for i in index_str.split(',')]
        except ValueError:
            print("[ERROR] Invalid input format for indices.")
            return None

        color_str = input("Enter list of new colors (e.g., #FF5733,#00FF00,#123456): ")
        target_colors = [c.strip() for c in color_str.split(',')]

        if len(index_list) != len(target_colors):
            print("[ERROR] Number of indices must match number of target colors.")
            return None

        color_mapping = dict(zip(index_list, target_colors))
        output_path = f"color_edited_{os.path.basename(image_path)}"
        modified_image = indexer.bulk_color_change(color_mapping, output_path)

        if modified_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(np.array(modified_image))
            plt.axis('off')
            plt.title('Color Edited Image')
            plt.show()
            print(f"[INFO] Edited image saved to {output_path}")

            export_to_psd(output_path, color_mapping, output_psd="output_layers.psd")
            print("[INFO] PSD file with editable layers has been created.")

        return modified_image

    except Exception as e:
        print(f"[ERROR] Color editing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# Main function to run the textile design generator

def main():
    print("=== Textile Design Generator ===")
    choice = input(
        "What do you want to do? "
        "(1) Create Image (2) Edit Image (3) Edit Colors (4) Tile Image: "
    )

    if choice == "1":
        # Generate a new image
        has_reference = input(
            "Do you want to upload a reference image? (yes/no): "
        ).lower() == 'yes'

        description = input("Enter image description: ")
        style = input("Enter style (standalone / repetitive pattern): ")
        color_info = input("Enter color requirement (number or range): ")

        try:
            simplicity = int(input("Enter simplicity level (1-10): "))
            simplicity = max(1, min(10, simplicity))
        except ValueError:
            print("[WARNING] Invalid simplicity level. Using default value of 5.")
            simplicity = 5

        try:
            n_options = int(input("How many design variations needed?: "))
            n_options = max(1, min(5, n_options))
        except ValueError:
            print("[WARNING] Invalid number of options. Using default value of 1.")
            n_options = 1

        prompt = create_prompt(description, style, color_info, simplicity)
        print(f"[INFO] Generated Prompt: {prompt}")

        if has_reference:
            reference_image_folder = input(
                "Enter the folder path containing reference images: "
            )
            if not os.path.isdir(reference_image_folder):
                print(f"[ERROR] Folder not found: {reference_image_folder}")
                return

            image_files = [
                os.path.join(reference_image_folder, f)
                for f in os.listdir(reference_image_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            if not image_files:
                print(
                    f"[WARNING] No image files found in {reference_image_folder}. "
                    "Generating without reference."
                )
                image_files = []

            if simplicity <= 5:
                images = generate_with_reference_gemini(prompt, image_files, n_options)
            else:
                images = generate_with_reference_dalle(prompt, image_files, n_options)
        else:
            if simplicity <= 5:
                images = generate_without_reference_gemini(prompt, n_options)
            else:
                images = generate_without_reference_dalle(prompt, n_options)

        if images:
            print(f"[INFO] Generated {len(images)} images.")
            for i, img in enumerate(images):
                plt.figure(figsize=(10, 10))
                plt.imshow(np.array(img))
                plt.axis('off')
                plt.title(f'Generated Image {i+1}')
                plt.show()
                # Save the single generated image
                output_path = f"single_image_{i}.png"
                img.save(output_path, dpi=(300, 300))
                print(f"[INFO] Saved single image to {output_path}")
                # Create a tiled version of the image
                tile_image_grid(output_path, output_path=f"tiled_image_{i}.png")

        else:
            print("[ERROR] No images generated.")

    elif choice == "2":
        # Edit an existing image
        description = input("Describe the changes you want: ")
        image_path = input("Enter the URL/path of the image: ")

        try:
            simplicity = int(input("Enter simplicity level (1-10): "))
            simplicity = max(1, min(10, simplicity))
        except ValueError:
            print("[WARNING] Invalid simplicity level. Using default value of 5.")
            simplicity = 5

        if not os.path.exists(image_path):
            print(f"[ERROR] Image file '{image_path}' not found.")
            return

        if simplicity <= 5:
            edited_image = edit_image_gemini(description, image_path)
        else:
            edited_image = edit_image_dalle(description, image_path)

        if edited_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(np.array(edited_image))
            plt.axis('off')
            plt.title('Edited Image')
            plt.show()
            # Save the edited image
            output_path = f"single_edited_image.png"
            edited_image.save(output_path, dpi=(300, 300))
            print(f"[INFO] Saved edited image to {output_path}")
            # Create a tiled version of the edited image
            tile_image_grid(output_path, output_path="tiled_edited_image.png")
        else:
            print("[ERROR] Image editing failed.")

    elif choice == "3":
        # Change colors in an existing image
        image_path = input("Enter image URL/path: ")
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file '{image_path}' not found.")
            return

        modified_image = edit_colors(image_path)
        if modified_image:
            # Save the color-edited image
            output_path = "single_color_edited_image.png"
            modified_image.save(output_path, dpi=(300, 300))
            print(f"[INFO] Saved color-edited image to {output_path}")
            # Create a tiled version of the color-edited image
            tile_image_grid(output_path, output_path="tiled_color_edited_image.png")
        else:
            print("[ERROR] Color editing failed.")

    elif choice == "4":
        # Tile an existing image
        image_path = input("Enter image URL/path: ")
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file '{image_path}' not found.")
            return

        try:
            grid_rows = int(input("Enter number of rows for the grid (default 5): ") or "5")
            grid_cols = int(input("Enter number of columns for the grid (default 5): ") or "5")
        except ValueError:
            print("[WARNING] Invalid grid dimensions. Using default 5x5 grid.")
            grid_rows, grid_cols = 5, 5

        tile_image_grid(image_path, grid_rows, grid_cols, output_path="tiled_image.png")

    else:
        print("[ERROR] Invalid choice. Please select 1, 2, 3, or 4.")

# =============================================================================
# # Run the program
# if __name__ == "__main__":
#     main()
# =============================================================================
