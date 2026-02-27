# Real-Time CCTV Surveillance with Vision-Language Models

A Streamlit-based surveillance application that analyzes uploaded CCTV video, generates frame captions using a vision-language model, classifies incident risk levels, plays audio alerts, and exports a structured PDF incident report.

## Features

- Video analysis from uploaded `.mp4` files
- Frame sampling at configurable intervals
- Caption generation using BLIP (`Salesforce/blip-image-captioning-base`)
- Rule-based incident detection:
  - `SAFE`
  - `SUSPICIOUS`
  - `DANGEROUS`
- Live UI with:
  - risk-colored frame borders
  - current caption
  - threat assessment
  - explanation panel
- Detection log table with timestamps
- Audio alert modes:
  - beep
  - custom `.wav`
- PDF export of incident report from detection logs

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- PyTorch
- Transformers (Hugging Face)
- Pillow

## Project Structure

```text
Final year/
  app/
    app2.py               # Main Streamlit app (current)
    main2.py              # Analysis pipeline
    incident_detector.py  # Rule-based risk classifier
    caption_generator.py  # BLIP caption generation
    frame_extractor.py    # Frame sampling from video
    audio_alert.py        # Alert sound handling
    report_exporter.py    # PDF report generation
  requirements.txt
  .gitignore
```

## Setup

1. Clone the repository:
```powershell
git clone <your-repo-url>
cd "<repo-folder>"
```

2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Run the App

From the project root:

```powershell
streamlit run app/app2.py
```

Then open the local Streamlit URL shown in terminal (usually `http://localhost:8501`).

## How It Works

1. Upload an MP4 file in the sidebar.
2. Set frame interval (lower value = denser analysis).
3. Start surveillance.
4. For each sampled frame:
   - caption is generated
   - risk level is classified
   - UI and logs are updated
   - audio alert is triggered (based on settings)
5. Download a PDF report from the Detection Log section.

## Notes

- Audio alerts use `winsound`, which is Windows-specific.
- Webcam input is currently marked as "Coming Soon".
- Model inference can be slow on CPU. GPU (CUDA) improves performance.

## Future Improvements

- WhatsApp/SMS notifications for critical incidents
- Webcam/live stream support
- Database-backed incident history
- Snapshot images in PDF reports

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
