# Real-Time CCTV Surveillance with Vision-Language Models

A Streamlit-based surveillance application that keeps the webcam feed live on screen while saving short clips in the background, analyzing them with a vision-language pipeline, logging caption and sentiment results, deleting safe clips, and retaining unsafe clips for review.

## Features

- Low-latency live webcam preview in the browser
- Continuous 2-3 second clip recording from the live feed
- Background clip analysis so the live view is not blocked by model inference
- Caption generation using BLIP (`Salesforce/blip-image-captioning-base`)
- Sentiment analysis on generated captions
- Rule-based incident detection:
  - `SAFE`
  - `SUSPICIOUS`
  - `DANGEROUS`
- Backend log table with:
  - timestamp
  - caption
  - sentiment
  - unsafe decision
  - SMS status
  - saved clip path
- Safe clips are deleted after processing
- Unsafe clips are moved into a separate folder
- Optional Windows audio alerts
- Optional SMS alerts through Twilio environment variables
- PDF export of incident logs

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
    sentiment_analyzer.py # Caption sentiment analysis
    frame_extractor.py    # Frame sampling from video
    audio_alert.py        # Alert sound handling
    sms_notifier.py       # Twilio SMS integration
    webcam_backend.py     # Live webcam capture + clip processing
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

1. Start the webcam in the sidebar.
2. The browser shows the live feed directly with minimal processing.
3. The backend writes short `.mp4` clips into a pending folder.
4. Each clip is analyzed in the background:
   - frames are sampled from the clip
   - a caption is generated
   - rule-based risk is computed
   - sentiment is computed from the caption
   - a backend log entry is created
5. If the clip is safe, it is deleted.
6. If the clip is unsafe, it is moved to the unsafe clips folder and an SMS can be sent.
7. Logs can be exported as a PDF report.

## Notes

- Audio alerts use `winsound`, which is Windows-specific.
- Model inference can be slow on CPU. GPU (CUDA) improves performance.
- SMS alerts require these environment variables:
  - `TWILIO_ACCOUNT_SID`
  - `TWILIO_AUTH_TOKEN`
  - `TWILIO_FROM_NUMBER`
  - `TWILIO_TO_NUMBER`

## Future Improvements

- Database-backed incident history
- Smarter clip-level video understanding models
- Dashboard for reviewing archived unsafe clips
- Snapshot images in PDF reports

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
