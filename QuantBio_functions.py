#Functions for RFCML training, inference, annotate_video plots

import cv2
import pandas as pd
import numpy as np
from collections import deque
import pandas as pd 
import os
import glob

#converting temporal segments to frames from the online annotator and combining ground truth data; can be removed before handing off to students?

def get_total_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def temporal_to_frame_labels(csv_path, total_frames, fps=60):

    df = pd.read_csv(csv_path)
    labels = np.zeros(total_frames, dtype=int)

    for segment in df["temporal_coordinates"]:

        # safer parsing instead of eval
        segment = segment.strip("[]")
        start_time, end_time = map(float, segment.split(","))

        start_frame = int(round(start_time * fps))
        end_frame   = int(round(end_time * fps))

        # clip to valid range
        start_frame = max(0, start_frame)
        end_frame   = min(total_frames - 1, end_frame)

        if start_frame <= end_frame:
            labels[start_frame:end_frame+1] = 1

    return labels


def combine_annotators_for_video(
    annotator_csvs,
    total_frames,
    fps=60,
    method="majority"   #if more than 50% annotations per frame was fight, final label would be fight
):

    all_labels = []

    for csv in annotator_csvs:
        labels = temporal_to_frame_labels(csv, total_frames, fps)
        all_labels.append(labels)

    all_labels = np.array(all_labels)

    if method == "majority":
        final_labels = (all_labels.mean(axis=0) >= 0.5).astype(int)

    elif method == "any":
        final_labels = (all_labels.sum(axis=0) > 0).astype(int)

    elif method == "all":
        final_labels = (all_labels.sum(axis=0) == len(annotator_csvs)).astype(int)

    else:
        raise ValueError("Invalid combination method.")

    return final_labels


def generate_final_label_file(
    annotator_csvs,
    video_path,
    output_path,
    fps=60,
    method="majority"
):

    total_frames = get_total_frames_from_video(video_path)

    print("Total frames detected:", total_frames)

    final_labels = combine_annotators_for_video(
        annotator_csvs,
        total_frames,
        fps=fps,
        method=method
    )

    df_out = pd.DataFrame({
        "frame": np.arange(total_frames),
        "label": final_labels
    })

    df_out.to_csv(output_path, index=False)

    print("Saved:", output_path)

#feature+temporal feature computations; can be hidden for students to try on their own?

def compute_basic_features(traj_df, fps=60):

    fish1 = traj_df.iloc[:, [1, 2]].values
    fish2 = traj_df.iloc[:, [3, 4]].values

    fish1 = np.nan_to_num(fish1, nan=0.0)
    fish2 = np.nan_to_num(fish2, nan=0.0)

    # Distance
    distance = np.linalg.norm(fish1 - fish2, axis=1)

    # Speed
    speed1 = np.linalg.norm(
        np.diff(fish1, axis=0, prepend=fish1[0:1]),
        axis=1
    ) * fps

    speed2 = np.linalg.norm(
        np.diff(fish2, axis=0, prepend=fish2[0:1]),
        axis=1
    ) * fps

    # Acceleration
    acc1 = np.diff(speed1, prepend=speed1[0]) * fps
    acc2 = np.diff(speed2, prepend=speed2[0]) * fps

    # Heading
    dx1 = np.diff(fish1[:, 0], prepend=fish1[0, 0])
    dy1 = np.diff(fish1[:, 1], prepend=fish1[0, 1])
    heading1 = np.arctan2(dy1, dx1)

    dx2 = np.diff(fish2[:, 0], prepend=fish2[0, 0])
    dy2 = np.diff(fish2[:, 1], prepend=fish2[0, 1])
    heading2 = np.arctan2(dy2, dx2)

    features = pd.DataFrame({
        "frame": np.arange(len(distance)),
        "inter_animal_distance": distance,
        "speed1": speed1,
        "speed2": speed2,
        "acc1": acc1,
        "acc2": acc2,
        "heading1": heading1,
        "heading2": heading2
    })

    return features

def generate_temporal_features(
    df,
    feature_columns,
    window_size=40
):

    data = df[feature_columns].to_numpy()
    frames = df["frame"].to_numpy()

    results = []
    n = len(df)

    for i in range(window_size - 1, n):

        win = data[i - window_size + 1 : i + 1]

        feats = {}
        for j, col in enumerate(feature_columns):
            x = win[:, j]

            feats[f"{col}_mean"] = x.mean()
            feats[f"{col}_std"]  = x.std()
            feats[f"{col}_max"]  = x.max()
            feats[f"{col}_min"]  = x.min()
            feats[f"{col}_delta"] = x[-1] - x[0]

        feats["frame"] = frames[i]
        results.append(feats)

    return pd.DataFrame(results)

#processes dataset according to folder structure and saves temporal features

def process_dataset_folder(
    folder_path,
    fps=60,
    window_size=40
):

    traj_files = glob.glob(os.path.join(folder_path, "*trajectories*.csv"))
    if not traj_files:
        raise ValueError("No trajectory file found.")
    traj_file = traj_files[0]

    label_files = glob.glob(os.path.join(folder_path, "*label*.csv"))
    has_labels = len(label_files) > 0
    label_file = label_files[0] if has_labels else None

    print("Trajectory:", traj_file)
    if has_labels:
        print("Labels:", label_file)
    else:
        print("No label file found — running in prediction mode.")

   
    traj_df = pd.read_csv(traj_file)

    
    features_df = compute_basic_features(traj_df, fps=fps)

   
    if has_labels:
        label_df = pd.read_csv(label_file)
        merged = pd.merge(features_df, label_df, on="frame", how="inner")
    else:
        merged = features_df

    feature_columns = [
        "inter_animal_distance",
        "speed1", "speed2",
        "acc1", "acc2",
        "heading1", "heading2"
    ]

   
    temporal_df = generate_temporal_features(
        merged,
        feature_columns,
        window_size=window_size
    )

    if has_labels:
        temporal_df = pd.merge(
            temporal_df,
            merged[["frame", "label"]],
            on="frame",
            how="left"
        )
        out_name = "temporal_features_with_labels.csv"
    else:
        out_name = "temporal_features.csv"

    out_path = os.path.join(folder_path, out_name)
    temporal_df.to_csv(out_path, index=False)

    print("Saved:", out_path)

    return out_path

#annoattion of 5 minute video chunk with most fight instances

def annotate_most_fight_chunk(
    video_path,
    trajectory_csv,
    predictions_csv,
    output_path,
    window_minutes=5,
    threshold=0.4,
    trail_length=10
):


    traj_df = pd.read_csv(trajectory_csv)
    pred_df = pd.read_csv(predictions_csv)

    print("Trajectory columns:", traj_df.columns.tolist())
    print("Prediction columns:", pred_df.columns.tolist())


    if "frame" not in traj_df.columns:
        if "time" in traj_df.columns:
            print("Converting trajectory time → frame (fps=60)")
            traj_df["frame"] = (traj_df["time"] * 60).round().astype(int)
        else:
            raise ValueError("Trajectory CSV missing both 'frame' and 'time' columns.")


    if "frame" not in pred_df.columns:
        raise ValueError("Predictions CSV missing 'frame' column.")


    df = pd.merge(traj_df, pred_df, on="frame", how="inner")
    df = df.sort_values("frame").reset_index(drop=True)



    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_size = int(window_minutes * 60 * fps)

    print("FPS:", fps)
    print("Window size (frames):", window_size)

    fight_array = (
        (df["fight_probability"] >= threshold) |
        (df["predicted_label"] == 1)
    ).astype(int).values

    if len(fight_array) < window_size:
        raise ValueError("Video shorter than window size.")

    fight_density = np.array([
        fight_array[i:i+window_size].sum()
        for i in range(len(fight_array) - window_size)
    ])

    best_start_idx = np.argmax(fight_density)
    best_start_frame = int(df.loc[best_start_idx, "frame"])
    best_end_frame = best_start_frame + window_size

    print("Most fight-dense window:")
    print("Start frame:", best_start_frame)
    print("End frame:", best_end_frame)


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trail1 = deque(maxlen=trail_length)
    trail2 = deque(maxlen=trail_length)

    cap.set(cv2.CAP_PROP_POS_FRAMES, best_start_frame)
    frame_idx = best_start_frame

    print("Starting annotation...")

    while cap.isOpened() and frame_idx < best_end_frame:

        ret, frame = cap.read()
        if not ret:
            break

        row = df[df["frame"] == frame_idx]

        if not row.empty:

            row = row.iloc[0]

            x1, y1 = row["x1"], row["y1"]
            x2, y2 = row["x2"], row["y2"]

            # Centroids
            if not np.isnan(x1) and not np.isnan(y1):
                pos1 = (int(x1), int(y1))
                cv2.circle(frame, pos1, 3, (0, 255, 255), -1)
                trail1.append(pos1)

            if not np.isnan(x2) and not np.isnan(y2):
                pos2 = (int(x2), int(y2))
                cv2.circle(frame, pos2, 3, (0, 255, 0), -1)
                trail2.append(pos2)

            # Trails
            for p in trail1:
                cv2.circle(frame, p, 1, (0, 255, 255), -1)

            for p in trail2:
                cv2.circle(frame, p, 1, (0, 255, 0), -1)

            # Fight overlay
            if row["fight_probability"] >= threshold or row["predicted_label"] == 1:
                cv2.putText(
                    frame,
                    "FIGHT",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    4
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print("Annotated chunk saved to:", output_path)
