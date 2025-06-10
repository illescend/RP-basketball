# # data_collection.py
#
# import os
# os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # Try to disable Media Foundation
# os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100"  # Try to prioritize FFMPEG
#
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from testmodel.algorithms import utils
#
# # --- Configuration ---
# # You can set these here or import them from a shared config file if you prefer
# BALL_CONF_THRESHOLD = 0.4
# POSE_CONF_THRESHOLD = 0.35
# TARGET_BALL_CLASS_ID = 0  # Assuming 0 is 'ball' in your custom model
#
# # --- Data Collection Parameters ---
# FRAMES_BEFORE = 5  # Number of frames to save before the heuristic release
# FRAMES_AFTER = 5  # Number of frames to save after the heuristic release
# SHOT_COOLDOWN_PERIOD = 200  # Frames to wait after detecting a shot to avoid duplicates
#
# # --- Filtering Parameters for "Actual Shots" ---
# MIN_BALL_HEIGHT_FOR_SHOT_GATHER = 0.45  # Ball's y-coord must be above 45% of frame height (from top)
# MIN_WRIST_HEIGHT_FOR_SHOT_GATHER = 0.55  # Wrist's y-coord must be above 55% of frame height
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#
#
# # --- Main Data Collection Pipeline ---
# def collect_release_candidates(video_path, output_base_dir):
#     """
#     Processes a video to find candidate shot releases and saves frame windows for labeling,
#     using functions from the utils.py module.
#     """
#     # --- Setup ---
#     print(f"Starting data collection for: {video_path}")
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     output_dir = os.path.join(output_base_dir, video_name)
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Saving candidate frames to: {output_dir}")
#
#     MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "Yolo-Weights")
#     BALL_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "model2.pt")
#     POSE_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "yolov8n-pose.pt")
#
#     ball_model = YOLO(BALL_MODEL_PATH)
#     pose_model = YOLO(POSE_MODEL_PATH)
#
#     # Video handling
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error opening video: {video_path}")
#         return
#
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # State variables
#     frame_count = 0
#     shot_candidate_count = 0
#     potential_release_buffer = []
#     cooldown_timer = 0
#     frame_buffer = []
#     buffer_size = FRAMES_BEFORE + FRAMES_AFTER + 10
#
#     # --- Main Loop ---
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_count += 1
#
#         # Maintain frame buffer
#         frame_buffer.append((frame_count, frame.copy()))
#         if len(frame_buffer) > buffer_size:
#             frame_buffer.pop(0)
#
#         # Handle cooldown
#         if cooldown_timer > 0:
#             cooldown_timer -= 1
#             continue
#
#         # --- Detections using utils ---
#         ball_results = ball_model(frame, conf=BALL_CONF_THRESHOLD, verbose=False, classes=[TARGET_BALL_CLASS_ID])
#         pose_results = pose_model(frame, conf=POSE_CONF_THRESHOLD, verbose=False)
#
#         ball_bbox = utils.get_ball_bbox(ball_results, TARGET_BALL_CLASS_ID)
#
#         # Use the advanced get_likely_shooter_wrist_and_person_idx from your utils
#         wrist_pos, person_idx = None, None  # Initialize
#         if ball_bbox is not None:
#             # Note: Your utils function returns (wrist_pos, person_idx). We'll use them.
#             wrist_pos, person_idx = utils.get_likely_shooter_wrist_and_person_idx(
#                 pose_results_list=pose_results,
#                 ball_bbox=ball_bbox,
#                 shooting_arm_wrist_idx=utils.RIGHT_WRIST,
#                 shooting_arm_elbow_idx=utils.RIGHT_ELBOW,
#                 shooting_arm_shoulder_idx=utils.RIGHT_SHOULDER,
#                 max_dist_threshold=utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
#             )
#
#         # --- Heuristic Logic ---
#         is_shot_stance = False
#         if ball_bbox is not None and wrist_pos is not None:
#             ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
#             if ball_center_y < frame_height * MIN_BALL_HEIGHT_FOR_SHOT_GATHER and \
#                     wrist_pos[1] < frame_height * MIN_WRIST_HEIGHT_FOR_SHOT_GATHER:
#                 is_shot_stance = True
#
#         if is_shot_stance:
#             wrist_in_ball = utils.is_point_inside_bbox(wrist_pos, ball_bbox, margin=utils.WRIST_BALL_PROXIMITY_MARGIN)
#
#             if not wrist_in_ball:
#                 potential_release_buffer.append(frame_count)
#                 if len(potential_release_buffer) > utils.CONFIRMATION_FRAMES_FOR_RELEASE:
#                     potential_release_buffer.pop(0)
#
#                 if len(potential_release_buffer) == utils.CONFIRMATION_FRAMES_FOR_RELEASE:
#                     heuristic_release_frame = potential_release_buffer[0]
#                     shot_candidate_count += 1
#                     print(
#                         f"\n>>> Candidate Shot #{shot_candidate_count} detected around frame {heuristic_release_frame} <<<")
#
#                     # Save the window of frames
#                     start_frame = heuristic_release_frame - FRAMES_BEFORE
#                     end_frame = heuristic_release_frame + FRAMES_AFTER
#
#                     saved_count = 0
#                     shot_folder = os.path.join(output_dir, f"shot_{shot_candidate_count:03d}")
#                     os.makedirs(shot_folder, exist_ok=True)  # Create a dedicated folder for this shot
#
#                     for f_num, f_img in frame_buffer:
#                         if start_frame <= f_num <= end_frame:
#                             save_path = os.path.join(shot_folder, f"frame_{f_num:05d}.png")
#                             cv2.imwrite(save_path, f_img)
#                             saved_count += 1
#
#                     print(f"Saved {saved_count} frames to folder '{os.path.basename(shot_folder)}'")
#
#                     potential_release_buffer.clear()
#                     cooldown_timer = SHOT_COOLDOWN_PERIOD
#             else:
#                 potential_release_buffer.clear()
#         else:
#             potential_release_buffer.clear()
#
#         # Optional: Display for debugging
#         # debug_frame = frame.copy()
#         # if ball_bbox is not None: cv2.rectangle(debug_frame, (ball_bbox[0], ball_bbox[1]), (ball_bbox[2], ball_bbox[3]), (0,255,0), 1)
#         # if wrist_pos is not None: cv2.circle(debug_frame, tuple(wrist_pos), 5, (0,0,255), -1)
#         # cv2.imshow("Data Collection", debug_frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#
#     # --- Cleanup ---
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"\nFinished processing. Found {shot_candidate_count} candidate shots.")
#
#
# if __name__ == '__main__':
#     VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "footage")
#     DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "release_model","release_frames")  # Changed output dir slightly
#     os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
#
#     video_path = os.path.join(VIDEO_INPUT_DIR, "GH012211.mp4")
#
#     collect_release_candidates(video_path, DATA_OUTPUT_DIR)

# data_collection.py (updated for Two-Stream Model Data)

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # Try to disable Media Foundation
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "100"  # Try to prioritize FFMPEG

import cv2
import numpy as np
from ultralytics import YOLO
from testmodel.algorithms import utils  # Assuming this is your utils module

# --- Configuration ---
BALL_CONF_THRESHOLD = 0.4
POSE_CONF_THRESHOLD = 0.35
TARGET_BALL_CLASS_ID = 0

# --- Data Collection Parameters ---
FRAMES_BEFORE = 10  # Number of frames to save before the heuristic release
FRAMES_AFTER = 5  # Number of frames to save after the heuristic release
SHOT_COOLDOWN_PERIOD = 200  # Frames to wait after detecting a shot

# NEW: Patch size for the cropped images
PATCH_SIZE = 96  # Pixels (e.g., 96x96). This will be the input size for your CNN.

# --- Filtering Parameters for "Actual Shots" ---
MIN_BALL_HEIGHT_FOR_SHOT_GATHER = 0.45
MIN_WRIST_HEIGHT_FOR_SHOT_GATHER = 0.55
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def crop_patch(full_frame, center_xy, patch_size):
    """
    Crops a square patch from a frame centered at a given point.
    Handles boundary conditions by padding with black.
    """
    if center_xy is None:
        return None

    cx, cy = center_xy
    half_size = patch_size // 2

    # Calculate crop coordinates
    x1, y1 = cx - half_size, cy - half_size
    x2, y2 = cx + half_size, cy + half_size

    # Create a black padded frame to handle out-of-bounds cropping
    h, w, _ = full_frame.shape
    padded_frame = cv2.copyMakeBorder(full_frame, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])

    # Get the patch from the padded frame
    patch = padded_frame[y1 + half_size: y2 + half_size, x1 + half_size: x2 + half_size]

    # Ensure the final patch is the correct size, resizing if necessary
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    return patch


# --- Main Data Collection Pipeline ---
def collect_release_candidates(video_path, output_base_dir):
    """
    Processes a video to find candidate shot releases and saves hand/ball patches.
    """
    # --- Setup ---
    print(f"Starting data collection for: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving candidate patches to: {output_dir}")

    MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "Yolo-Weights")
    BALL_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "model2.pt")
    POSE_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "yolov8n-pose.pt")

    ball_model = YOLO(BALL_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # State variables
    frame_count = 0
    shot_candidate_count = 0
    potential_release_buffer = []
    cooldown_timer = 0
    frame_buffer = []
    buffer_size = FRAMES_BEFORE + FRAMES_AFTER + 10

    # --- Main Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame_buffer.append((frame_count, frame.copy()))
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        if cooldown_timer > 0:
            cooldown_timer -= 1
            continue

        # --- Detections using utils ---
        ball_results = ball_model(frame, conf=BALL_CONF_THRESHOLD, verbose=False, classes=[TARGET_BALL_CLASS_ID])
        pose_results = pose_model(frame, conf=POSE_CONF_THRESHOLD, verbose=False)

        ball_bbox = utils.get_ball_bbox(ball_results, TARGET_BALL_CLASS_ID)

        wrist_pos = None
        if ball_bbox is not None:
            wrist_pos, _ = utils.get_likely_shooter_wrist_and_person_idx(
                pose_results_list=pose_results, ball_bbox=ball_bbox,
                shooting_arm_wrist_idx=utils.RIGHT_WRIST, shooting_arm_elbow_idx=utils.RIGHT_ELBOW,
                shooting_arm_shoulder_idx=utils.RIGHT_SHOULDER,
                max_dist_threshold=utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
            )

        # --- Heuristic Logic ---
        is_shot_stance = False
        if ball_bbox is not None and wrist_pos is not None:
            ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
            if ball_center_y < frame_height * MIN_BALL_HEIGHT_FOR_SHOT_GATHER and \
                    wrist_pos[1] < frame_height * MIN_WRIST_HEIGHT_FOR_SHOT_GATHER:
                is_shot_stance = True

        if is_shot_stance:
            wrist_in_ball = utils.is_point_inside_bbox(wrist_pos, ball_bbox, margin=utils.WRIST_BALL_PROXIMITY_MARGIN)

            if not wrist_in_ball:
                potential_release_buffer.append(frame_count)
                if len(potential_release_buffer) > utils.CONFIRMATION_FRAMES_FOR_RELEASE:
                    potential_release_buffer.pop(0)

                if len(potential_release_buffer) == utils.CONFIRMATION_FRAMES_FOR_RELEASE:
                    heuristic_release_frame = potential_release_buffer[0]
                    shot_candidate_count += 1
                    print(
                        f"\n>>> Candidate Shot #{shot_candidate_count} detected around frame {heuristic_release_frame} <<<")

                    # --- START OF NEW PATCH EXTRACTION LOGIC ---
                    start_frame_to_process = heuristic_release_frame - FRAMES_BEFORE
                    end_frame_to_process = heuristic_release_frame + FRAMES_AFTER

                    shot_folder = os.path.join(output_dir, f"shot_{shot_candidate_count:03d}")
                    os.makedirs(shot_folder, exist_ok=True)

                    saved_patch_count = 0
                    for f_num, f_img in frame_buffer:
                        if start_frame_to_process <= f_num <= end_frame_to_process:
                            # Re-run detection on this specific frame to get precise locations
                            temp_ball_results = ball_model(f_img, conf=BALL_CONF_THRESHOLD, verbose=False,
                                                           classes=[TARGET_BALL_CLASS_ID])
                            temp_pose_results = pose_model(f_img, conf=POSE_CONF_THRESHOLD, verbose=False)

                            temp_ball_bbox = utils.get_ball_bbox(temp_ball_results, TARGET_BALL_CLASS_ID)
                            temp_wrist_pos = None
                            if temp_ball_bbox is not None:
                                temp_wrist_pos, _ = utils.get_likely_shooter_wrist_and_person_idx(
                                    pose_results_list=temp_pose_results, ball_bbox=temp_ball_bbox,
                                    shooting_arm_wrist_idx=utils.RIGHT_WRIST, shooting_arm_elbow_idx=utils.RIGHT_ELBOW,
                                    shooting_arm_shoulder_idx=utils.RIGHT_SHOULDER,
                                    max_dist_threshold=utils.MAX_WRIST_BALL_DISTANCE_FOR_CONSIDERATION
                                )

                            # If both are found, crop and save the patches
                            if temp_ball_bbox is not None and temp_wrist_pos is not None:
                                ball_center = (int((temp_ball_bbox[0] + temp_ball_bbox[2]) / 2),
                                               int((temp_ball_bbox[1] + temp_ball_bbox[3]) / 2))

                                ball_patch = crop_patch(f_img, ball_center, PATCH_SIZE)
                                hand_patch = crop_patch(f_img, temp_wrist_pos, PATCH_SIZE)

                                if ball_patch is not None and hand_patch is not None:
                                    ball_save_path = os.path.join(shot_folder, f"frame_{f_num:05d}_ball.png")
                                    hand_save_path = os.path.join(shot_folder, f"frame_{f_num:05d}_hand.png")
                                    cv2.imwrite(ball_save_path, ball_patch)
                                    cv2.imwrite(hand_save_path, hand_patch)
                                    saved_patch_count += 1

                    print(
                        f"Saved {saved_patch_count} pairs of hand/ball patches to folder '{os.path.basename(shot_folder)}'")
                    # --- END OF NEW PATCH EXTRACTION LOGIC ---

                    potential_release_buffer.clear()
                    cooldown_timer = SHOT_COOLDOWN_PERIOD
            else:
                potential_release_buffer.clear()
        else:
            potential_release_buffer.clear()

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing. Found {shot_candidate_count} candidate shots.")


if __name__ == '__main__':
    VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "footage")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "algorithms", "release_model", "release_patches")  # New output dir name
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(VIDEO_INPUT_DIR, "GH012211.mp4")

    collect_release_candidates(video_path, DATA_OUTPUT_DIR)