import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
import os
import time

# --- Helper Functions for Biomechanical Calculations ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points (in degrees)."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_vertical_angle(a, b):
    """Calculates the angle of a line with respect to the vertical axis."""
    a = np.array(a)
    b = np.array(b)
    vertical_vector = np.array([0, -1]) # A vector pointing straight up
    line_vector = b - a
    
    # Normalize vectors
    unit_vertical = vertical_vector / np.linalg.norm(vertical_vector)
    unit_line = line_vector / np.linalg.norm(line_vector)
    
    dot_product = np.dot(unit_line, unit_vertical)
    angle = np.arccos(dot_product) * 180.0 / np.pi
    return angle

# --- Main Analysis Function ---

def analyze_cricket_shot(video_path):
    """
    Processes a cricket video to perform pose estimation, calculate biomechanics,
    and generate an annotated video and a final evaluation report.
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- Video I/O ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'annotated_video.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # --- Data Storage for Final Evaluation ---
    metrics_log = {
        'elbow_angles': [],
        'spine_leans': [],
        'head_knee_alignments': []
    }
    
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # --- Pose Estimation ---
        results = pose.process(image)

        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Extract Keypoints and Calculate Metrics ---
        try:
            landmarks = results.pose_landmarks.landmark
            
            # --- Get coordinates for relevant joints ---
            # Assuming a right-handed batsman for front/back identification
            # Front side is the left side of the body in the image (player facing right)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # --- 1. Front Elbow Angle ---
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            metrics_log['elbow_angles'].append(elbow_angle)

            # --- 2. Spine Lean ---
            mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            mid_shoulder = ((left_shoulder[0] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2, 
                            (left_shoulder[1] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2)
            spine_lean = calculate_vertical_angle(mid_hip, mid_shoulder)
            # Adjust lean direction (e.g., > 90 is leaning forward)
            if mid_shoulder[0] < mid_hip[0]:
                spine_lean = 180 - spine_lean
            metrics_log['spine_leans'].append(spine_lean)

            # --- 3. Head-over-Knee Vertical Alignment ---
            head_x = nose[0] * frame_width
            front_knee_x = left_knee[0] * frame_width
            head_knee_alignment = abs(head_x - front_knee_x)
            metrics_log['head_knee_alignments'].append(head_knee_alignment)

            # --- Live Overlays ---
            # Draw pose skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            # Display real-time metric readouts
            cv2.putText(image, f"Elbow Angle: {int(elbow_angle)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Spine Lean: {int(spine_lean)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Head Align (px): {int(head_knee_alignment)}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display short feedback cues
            feedback_y_pos = 150
            if elbow_angle > 90:
                cv2.putText(image, "Good elbow elevation", (10, feedback_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            if head_knee_alignment < 50: # Threshold in pixels, may need tuning
                cv2.putText(image, "Head over front knee", (10, feedback_y_pos + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Head not over front knee", (10, feedback_y_pos + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        except Exception as e:
            # Gracefully handle frames with no detected person
            # You could add a text overlay here like "Player not detected"
            pass
        
        # Write the annotated frame
        out.write(image)
        frame_count += 1

    # --- Final Processing and Cleanup ---
    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time
    print(f"Video processing completed.")
    print(f"Total frames: {frame_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()

    # --- Final Shot Evaluation ---
    evaluation = generate_final_evaluation(metrics_log, frame_width)
    evaluation_path = os.path.join(output_dir, 'evaluation.json')
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation, f, indent=4)

    print(f"Annotated video saved to: {output_video_path}")
    print(f"Evaluation report saved to: {evaluation_path}")

def generate_final_evaluation(metrics, frame_width):
    """Computes final scores and feedback based on logged metrics."""
    
    # --- Scoring Logic (Heuristics-based) ---
    # These thresholds are examples and should be tuned based on expert knowledge.
    
    # 1. Swing Control (based on elbow angle consistency and peak)
    max_elbow_angle = max(metrics['elbow_angles']) if metrics['elbow_angles'] else 0
    swing_score = min(10, (max_elbow_angle / 140) * 10) # Score out of 10, maxing out at 140 degrees
    swing_feedback = f"Achieved a peak front elbow angle of {int(max_elbow_angle)} degrees. Aim for 120-150 degrees for a full extension."
    if swing_score < 6:
        swing_feedback += " Focus on extending your front arm more through the shot."

    # 2. Head Position (based on head-knee alignment)
    # Calculate percentage of frames where head was aligned
    alignment_threshold = 0.1 * frame_width  # 10% of frame width as tolerance
    aligned_frames = [d for d in metrics['head_knee_alignments'] if d < alignment_threshold]
    head_pos_score = (len(aligned_frames) / len(metrics['head_knee_alignments'])) * 10 if metrics['head_knee_alignments'] else 0
    head_pos_feedback = f"Your head was positioned correctly over your front knee for {int((head_pos_score/10)*100)}% of the shot."
    if head_pos_score < 7:
        head_pos_feedback += " Try to keep your head leaning forward and steady over your knee to maintain balance and power."

    # 3. Balance (based on spine lean)
    # Ideal spine lean is slightly forward, e.g., 70-85 degrees.
    avg_spine_lean = np.mean(metrics['spine_leans']) if metrics['spine_leans'] else 90
    balance_score = 10 - (abs(avg_spine_lean - 80) / 10) * 10 # Penalize deviation from 80 degrees
    balance_score = max(0, balance_score)
    balance_feedback = f"Your average spine lean was {int(avg_spine_lean)} degrees. A slight forward lean (around 75-85 degrees) is ideal."
    if balance_score < 7:
        balance_feedback += " Avoid falling away or being too upright; a stable base is key."
    
    # 4. Footwork (Placeholder as we don't track foot direction in detail)
    footwork_score = 7 # Placeholder score
    footwork_feedback = "Footwork analysis requires more specific metrics (like foot angle). As a general tip, ensure a stable base with your front foot pointing towards the cover region."

    # --- Compile the report ---
    report = {
        "summary_scores": {
            "Footwork": round(footwork_score, 1),
            "Head Position": round(head_pos_score, 1),
            "Swing Control": round(swing_score, 1),
            "Balance": round(balance_score, 1)
        },
        "detailed_feedback": [
            {
                "category": "Swing Control",
                "score": round(swing_score, 1),
                "feedback": swing_feedback
            },
            {
                "category": "Head Position",
                "score": round(head_pos_score, 1),
                "feedback": head_pos_feedback
            },
            {
                "category": "Balance",
                "score": round(balance_score, 1),
                "feedback": balance_feedback
            },
            {
                "category": "Footwork",
                "score": round(footwork_score, 1),
                "feedback": footwork_feedback
            }
        ]
    }
    return report

# --- Script Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cricket Shot Analysis Tool")
    parser.add_argument("--input_video_path", type=str, required=True, help="Path to the input cricket video.")
    args = parser.parse_args()
    
    analyze_cricket_shot(args.input_video_path)