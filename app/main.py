from frame_extractor import FrameExtractor
from caption_generator import CaptionGenerator
from sentiment_analyzer import SentimentAnalyzer
from rich.console import Console
from smol_vlm import SMOLCaptionGenerator
from incident_detector import IncidentDetector

import os 
import sys
import winsound 
import time

def main():
    # --- Step 1: Extract frames from the video ---
    print("💡 Initiated 💡")

    video_path = "video/car_crash.mp4"          # your video
    output_dir = "frames_interval"            # folder to save frames
    frame_interval = 10           
       # change to 1 if you want ALL frames
    extractor = FrameExtractor(video_path,output_dir,frame_interval) 

    # seentimenter=SentimentAnalyzer()

    detector=IncidentDetector()

    frame_paths = extractor.extract_frames()  # list of saved frame paths  ------ #GPT edit

    console=Console()
    # caption_generator_smol=SMOLCaptionGenerator()

   # --- Step 2: Generate captions for each frame ---
    generator = CaptionGenerator(model_name="Salesforce/blip-image-captioning-base")    
    print("\n--- Generated Captions ---")

    for frame_path in frame_paths:

        caption = generator.generate_caption(frame_path,context="")

        result=detector.detect(caption)
        risk_level = result["risk_level"]
        risk_score = result["risk_score"]
        explanation = result["explanation"]

        
        # label,score=seentimenter.analyze(caption) # "res" ki jagah pe "caption" kr dena 
        console.print(
            f"[bold magenta]{frame_path}[/bold magenta]\n"
            f"[italic green]{caption}[/italic green]"
        )

        # --- Color mapping ---
        if risk_level == "DANGEROUS":
            color = "red"
        elif risk_level == "SUSPICIOUS":
            color = "yellow"
        else:
            color = "green"
            
        console.print(
            f"Risk Level : [{color}]{risk_level}[/{color}]"
            f"(score: {risk_score})"
        )
        console.print(f"[dim]{explanation}[/dim]\n")

        if risk_level=="DANGEROUS":
            console.print("[bold red]🚨 CRITICAL ALERT 🚨[/bold red]")
            for _ in range(2):
                winsound.Beep(2200, 300)
                winsound.Beep(1800, 450)
        elif risk_level=="SUSPICIOUS":
            console.print("[bold yellow]⚠️ Suspicious activity detected[/bold yellow]\n")

        # console.print(f"[bold magenta]{frame_path} [/bold magenta] → [italic green]{caption}[/italic green]")  # res ki jagah pe caption daal dio 
        # if label.lower() == "positive":
        #     color = "green"
        # elif label.lower() == "negative":
        #     color = "red"
        # else:  # neutral
        #     color = "deep_sky_blue4"
        # console.print(f"Sentiment:[{color}]{label}[/{color}] → (confidence:{score:.2f})\n")
        # if label.lower()=="negative":
        #     console.print("[bold red]⚠️  Alert  ⚠️[/bold red]")
        #     for _ in range(2):
        #         winsound.Beep(2200, 300)
        #         winsound.Beep(1800, 450)
    for file in os.listdir(output_dir):
        file_path=os.path.join(output_dir,file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("🚫 TERMINATED 🚫")
if __name__ == "__main__":
    main()
