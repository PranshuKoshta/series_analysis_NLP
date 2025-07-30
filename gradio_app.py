# gradio_app.py

import gradio as gr
from theme_classifier import ThemeClassifier
import pandas as pd
import plotly.express as px  # Import plotly

# Load the model ONCE when the app starts
print("Loading model, please wait...")
classifier = ThemeClassifier()
print("Model ready!")

def get_themes(theme_list_str, subtitles_path, save_path):

    theme_list = [theme.strip() for theme in theme_list_str.split(',')]
    
    # Use the global classifier instance
    output_df = classifier.get_themes(subtitles_path, theme_list, save_path)
    print("Classifier finished processing.")

    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]
    
    if not theme_list:
        print("No valid themes found to plot.")
        # Return an empty plot figure
        return px.bar(title="No Themes Found")

    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']
    print("Processing finished. Creating Plotly chart.")
    
    output_chart = px.bar(
        output_df,
        x='Score',
        y='Theme',
        orientation='h',  # 'h' for horizontal bars
        title="Series Themes",
        labels={'Score': 'Total Score', 'Theme': 'Identified Theme'}
    )
    return output_chart


def main():
    with gr.Blocks() as iface:
        gr.HTML("<h1>Theme Classification (Zero Shot Claasifiers)</h1>")
        with gr.Row():
            with gr.Column():
                # CHANGED: Use gr.Plot to display the chart
                plot = gr.Plot()
            with gr.Column():
                theme_list = gr.Textbox(label="Themes")
                subtitles_path = gr.Textbox(label="Subtitles or script Path")
                save_path = gr.Textbox(label="Save Path (Optional)")
                get_themes_button = gr.Button("Get Themes")
                get_themes_button.click(
                    get_themes, 
                    inputs=[theme_list, subtitles_path, save_path], 
                    outputs=[plot]
                )
    
    iface.queue().launch(share=True)

if __name__ == "__main__":
    main()