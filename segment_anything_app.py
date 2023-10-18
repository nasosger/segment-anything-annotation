""" Author: Anastasios Gerontopoulos, ECE AUTH
    In this Python script, we attempt to create an interactive app, with its own UI using gradio module. 
    The app's goal is to create binary segmentation masks, with META AI latest release, Segment Anything Model.
"""

import gradio as gr
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
from PIL import Image
import cv2



device = "cuda:0"                                       # GPU
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"         # model checkpoint (default weights)
model_type = "vit_h"                                    # Vision transformer as SAM encoder
torch.cuda.empty_cache()
model = sam_model_registry[model_type](checkpoint=sam_checkpoint)   # Define the model and push to GPU
model.to(device)

predictor = SamPredictor(sam_model=model)               # Class to predict the masks given interactive prompts.

selected_pixels = []

# Using the Blocks API for more flexibility.

with gr.Blocks() as demo:
    
    gr.Markdown("Segment Anything Model for Annotation")        # Title of the interface
    with gr.Row():
        input_image = gr.Image(label="Input RGB")               # Default mode: Numpy array, (W, H, C)
        output_mask = gr.Image(label="Output binary Mask")      # Same here

    with gr.Row():
        embed_button = gr.Button("Embed Image")
        embedding_state = gr.Textbox(label="Embedding State")
        filename = gr.Textbox(label="Filepath")         # Define the filename for saving
        save_button = gr.Button("Save mask")                                    # Button to save the prediction

    
    def generate_mask(image , evt: gr.SelectData):                              
        selected_pixels.append(evt.index)                       # Gather the selected pixels

        input_points = np.array(selected_pixels)
        # print(input_points)
        input_label = np.ones(input_points.shape[0])
        
        mask, _, _ = predictor.predict(point_coords=input_points, point_labels=input_label, multimask_output=False)
        # This predict function outputs the mask (C, H, W) and the confidence level (C,)
        mask  = Image.fromarray(mask[0, :, :])                              # mask has (1, sz, sz) output

        return mask


    def save_mask(mask, filename):
        mask = mask[:, :, 0]
        filename = filename + "Ids.png"
        cv2.imwrite(filename, mask)
        print(mask.shape)
        return

    def embed_image(image):
        global selected_pixels                                    # set selected pixels list to global
        selected_pixels = []                                      # empty the list to try again
        print(image.shape)
        predictor.set_image(image)                                # Set image for the embedding. Expects (H, W, C)
        state = "Embedding Complete"

        return state

    embed_button.click(embed_image, inputs=[input_image], outputs=[embedding_state])
    input_image.select(generate_mask, inputs=[input_image], outputs=[output_mask])
    save_button.click(save_mask, inputs=[output_mask, filename], outputs=[])


if __name__ == "__main__":
    demo.launch()
