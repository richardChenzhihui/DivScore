import gradio as gr
from divscore import DivScore
import torch
import os

# Set environment variables for Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the DivScore detector with loading state
def load_model():
    try:
        detector = DivScore(
            generalLM_name_or_path="mistral-community/Mistral-7B-v0.2",
            enhancedLM_name_or_path="RichardChenZH/DivScore_combined",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_bfloat16=True  # Use bfloat16 for better memory efficiency
        )
        return detector
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Global variable for the detector
detector = None

def detect_ai_text(text):
    """
    Detect if the input text is AI-generated using DivScore.
    Returns a tuple of (score, is_ai_generated, confidence)
    """
    global detector
    
    # Initialize detector if not already done
    if detector is None:
        detector = load_model()
        if detector is None:
            return "Error: Failed to load the model. Please try again later.", False, 0.0
    
    if not text.strip():
        return "Please enter some text to analyze.", False, 0.0
    
    try:
        score, entropy_score, ce_score = detector.compute_score(text)
        
        # Based on the paper's findings, we use 0.15 as the threshold
        is_ai_generated = score < 0.15

        result = f"DivScore: {score:.4f}\nEntropy Score: {entropy_score:.4f}\nCE Score: {ce_score:.4f}"
        return result, is_ai_generated
        
    except Exception as e:
        return f"Error occurred: {str(e)}", False, 0.0

# Create the Gradio interface with loading state
with gr.Blocks(title="DivScore AI Text Detector") as demo:
    gr.Markdown("""
    # DivScore AI Text Detector
    
    This demo uses the DivScore model to detect if text was generated by an AI model.
    Enter your text below to analyze it.
    
    **Note:** The model may take a few moments to load on first use.
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to analyze...",
                lines=5
            )
            submit_btn = gr.Button("Analyze Text")
        
        with gr.Column():
            result_output = gr.Textbox(label="Analysis Results")
            ai_generated = gr.Checkbox(label="AI Generated", interactive=False)
    
    gr.Examples(
        examples=[
            ["The quick brown fox jumps over the lazy dog."],
            ["Based on the analysis of the data, we can conclude that the implementation of the new protocol has resulted in a statistically significant improvement in patient outcomes."]
        ],
        inputs=text_input
    )
    
    submit_btn.click(
        fn=detect_ai_text,
        inputs=text_input,
        outputs=[result_output, ai_generated]
    )

if __name__ == "__main__":
    demo.queue()  # Enable queuing for better handling of multiple requests
    demo.launch() 