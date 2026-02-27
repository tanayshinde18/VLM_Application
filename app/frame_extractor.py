import cv2
import os
class FrameExtractor:
    def __init__(self, video_path,output_dir="frames", frame_interval=1):  
        """
        :param video_path: Path to the input video file
        :param output_dir: Directory where frames will be saved
        :param frame_interval: Save every nth frame (default=1 = all frames)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_interval = frame_interval

        # Make sure output dir exists
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file at {self.video_path}")
        frame_count = 0
        saved_count = 0
        saved_paths = []    

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % self.frame_interval == 0:
                frame_filename = os.path.join(self.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_paths.append(frame_filename)
                
                saved_count += 1

            frame_count += 1

        cap.release()

        print(f"\nProcessed {frame_count} frames, saved {saved_count} frames → {self.output_dir}")
        return saved_paths  # return list of saved frame file paths

        

