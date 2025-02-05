import gradio as gr
from transformers import pipeline

# Load the model
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def generate_caption(image):
    try:
        out = pipe(image)
        return out[0].get('generated_text', "No caption generated.")
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type='pil'),
    outputs="text",
    title="Image Captioning",
    description="Upload an image, and the model will generate a caption."
)

iface.launch()
