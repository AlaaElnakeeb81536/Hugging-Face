import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Hugging Face token not found in environment variables.")

# Load model and tokenizer with authentication
MODEL_NAME = "Lolity/results"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN).to(device)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Load dataset
try:
    df = pd.read_csv('Coursess.csv')
    print("Dataset loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading dataset: {str(e)}")

# Preprocessing steps
df['Track'] = df['Track'].str.strip().str.lower()
df_exploded = df.assign(Track=df['Track'].str.split(' / ')).explode('Track')
df_exploded['Track'] = df_exploded['Track'].str.strip().str.lower()

# Function to recommend courses
def recommend_courses(track, difficulty_level):
    target_track = track.strip().lower()
    target_level = difficulty_level.strip().lower()

    filtered_courses = df_exploded[
        (df_exploded["Track"] == target_track) & 
        (df_exploded["Difficulty Level"].str.strip().str.lower() == target_level)
    ]

    if filtered_courses.empty:
        return "No courses found for the given criteria."

    # Prepare a batch of course texts
    course_texts = [
        f"Track: {row['Track']}. Difficulty Level: {row['Difficulty Level']}. Description: {row['Course Description']}"
        for _, row in filtered_courses.iterrows()
    ]

    # Tokenize the batch
    inputs = tokenizer(course_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get scores for the batch
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.max(dim=1).values.cpu().numpy()

    # Combine course names and scores
    course_scores = list(zip(filtered_courses["Course Name"], scores))

    # Sort courses by score and select the top 5
    sorted_courses = sorted(course_scores, key=lambda x: x[1], reverse=True)
    top_courses = [name for name, _ in sorted_courses[:5]]

    # Format the output for Gradio
    output = f"Top courses for {track} ({difficulty_level}):\n\n"
    for course in top_courses:
        output += f"- {course}\n"

    return output

# Gradio Interface
iface = gr.Interface(
    fn=recommend_courses,
    inputs=[
        gr.Textbox(label="Track", placeholder="Enter the track (e.g., programming)"),
        gr.Dropdown(label="Difficulty Level", choices=["Beginner", "Intermediate", "Advanced"])
    ],
    outputs=gr.Textbox(label="Recommended Courses"),
    title="Course Recommendation System",
    description="Enter a track and difficulty level to get course recommendations."
)

# Launch the Gradio app
iface.launch()