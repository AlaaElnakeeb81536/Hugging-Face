import gradio as gr
from transformers import pipeline


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

'''
def summarize_text(text, max_length=130, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, clean_up_tokenization_spaces=True)
    return summary[0]['summary_text']
'''
# using Chunks
# Define the summarization function with chunking and batch processing
def summarize_text(text, max_length=130, min_length=30, chunk_size=512):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = summarizer(chunks, max_length=max_length, min_length=min_length, clean_up_tokenization_spaces=True)
    return " ".join([s['summary_text'] for s in summaries])

iface = gr.Interface(
    fn=summarize_text,  
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter text to summarize..."),  
        gr.Slider(minimum=30, maximum=200, value=130, label="Max Length"),  
        gr.Slider(minimum=10, maximum=100, value=30, label="Min Length")   
    ],
    outputs="text", 
    title="Text Summarization",
    description="Summarize long texts using the BART model from Hugging Face."
)

iface.launch()
