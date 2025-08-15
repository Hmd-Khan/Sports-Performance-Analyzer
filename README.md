## Quick Start: How to Run with Your Video

**Example command:**

```bash
python cover_drive_analysis_realtime.py --input_video_path input/Stunning_cover_drive_cricket_batreview.mp4
```

Replace `Stunning_cover_drive_cricket_batreview.mp4` with your own video filename if needed.

# Real-Time Cricket Shot Analysis System

## Quick Start: How to Run with Your Video

1. Place your cricket video file (e.g., `my_shot.mp4`) in the `input/` folder of this project.
2. Open a terminal in the project directory and activate your virtual environment (if not already active).
3. Run the following command, replacing `my_shot.mp4` with your actual video filename:
	```bash
	python cover_drive_analysis_realtime.py --input_video_path input/my_shot.mp4
	```
4. The annotated video and evaluation report will be saved in the `output/` folder.

# Real-Time Cricket Shot Analysis System

This project is a Python-based tool that analyzes cricket shots (such as the cover drive) from video. It uses pose estimation to provide real-time biomechanical feedback and generates an annotated video and a detailed performance evaluation.

---

## Features

- **Frame-by-Frame Video Processing:**
	- Reads and processes any cricket shot video using OpenCV.
- **Pose Estimation:**
	- Uses MediaPipe to extract 33 body keypoints for each frame.
- **Biomechanical Metrics:**
	- Calculates:
		- Front Elbow Angle
		- Spine Lean Angle
		- Head-Over-Knee Alignment
- **Live Overlays:**
	- Draws the pose skeleton and displays real-time metric values on the video.
	- Shows instant feedback (e.g., "Good elbow elevation", "Head not over front knee").
- **Final Evaluation:**
	- Computes scores (1-10) for Footwork, Head Position, Swing Control, and Balance.
	- Provides actionable feedback for each category.
	- Saves results to `output/evaluation.json`.

---

## Setup & Installation

1. **Clone or Download the Project**
	 - Download the project files or clone the repository to your computer.

2. **Install Python (Recommended: 3.10, 3.11, or 3.12)**
	 - Download from [python.org](https://www.python.org/downloads/).
	 - Add Python to your PATH during installation.

3. **Create a Virtual Environment**
	 - Open a terminal in the project folder and run:
		 ```bash
		 python -m venv .venv
		 # On Windows:
		 .venv\Scripts\activate
		 # On Mac/Linux:
		 source .venv/bin/activate
		 ```

4. **Install Required Libraries**
	 - With the virtual environment activated, run:
		 ```bash
		 pip install -r requirements.txt
		 ```
	 - If you see errors about missing packages (e.g., cv2), install them individually:
		 ```bash
		 pip install opencv-python mediapipe numpy
		 ```

---

## How to Use

1. **Prepare Your Video**
	 - Place your cricket shot video (e.g., `my_shot.mp4`) in the `input/` directory.

2. **Run the Analysis**
	 - In the terminal, run:
		 ```bash
		 python cover_drive_analysis_realtime.py --input_video_path input/my_shot.mp4
		 ```
	 - Replace `my_shot.mp4` with your actual video filename.

3. **View the Results**
	 - The annotated video will be saved as `output/annotated_video.mp4`.
	 - The performance evaluation will be saved as `output/evaluation.json`.

---

## Output Files

- `output/annotated_video.mp4`: Video with pose skeleton and live feedback overlays.
- `output/evaluation.json`: JSON file with scores and detailed feedback for each biomechanical category.

---

## Troubleshooting

- **cv2 or mediapipe not found:**
	- Run `pip install opencv-python mediapipe` in your virtual environment.
- **Script asks for --input_video_path:**
	- You must provide the path to your video file as a command-line argument.
- **Unicode symbols (✅, ❌) show as ??? in video:**
	- The script now uses plain English text for feedback overlays to ensure compatibility.
- **No output video or JSON:**
	- Check that your input video path is correct and the video is readable.
- **Multiple people in frame:**
	- The script is designed for a single player; results may be inaccurate with multiple people.

---

## Assumptions & Limitations

- **Camera Angle:**
	- Works best with side-on or front-on views. Extreme angles may reduce accuracy.
- **Single Player:**
	- Designed to track the most prominent person in the frame.
- **Player Orientation:**
	- Assumes a right-handed batsman by default. Adjust code for left-handed players if needed.
- **No Bat/Ball Tracking:**
	- Only analyzes body mechanics, not bat or ball movement.

---

## Customization & Extension

- You can adjust thresholds and feedback logic in the script to better fit your needs or expert recommendations.
- For left-handed batsmen or other shot types, modify the joint selection logic in the code.
- To add new metrics (e.g., bat angle, footwork), extend the pose analysis section.

---

## Support

If you have issues or suggestions, please open an issue or contact the project maintainer.
