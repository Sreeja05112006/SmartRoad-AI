import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import process_video


st.set_page_config(
    page_title="SmartRoad AI",
    page_icon="🚦",
    layout="wide",
)

st.title("SmartRoad AI")
st.subheader("Context-Aware Object Detection and Tracking for Road Safety")

st.write(
    "Upload a road video, compare a ByteTrack baseline with a "
    "context-aware tracker, and download the annotated video and analytics."
)


def convert_for_browser(input_path, output_path):
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return False

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )

        return (
            completed.returncode == 0
            and os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        )
    except (subprocess.SubprocessError, OSError):
        return False


def make_zones(width, height, enabled):
    if not enabled:
        return []

    return [
        {
            "name": "Lane A",
            "points": [
                [0, int(height * 0.45)],
                [width, int(height * 0.45)],
                [width, int(height * 0.68)],
                [0, int(height * 0.68)],
            ],
        },
        {
            "name": "Lane B",
            "points": [
                [0, int(height * 0.68)],
                [width, int(height * 0.68)],
                [width, height - 1],
                [0, height - 1],
            ],
        },
    ]


with st.sidebar:
    st.header("Analysis settings")

    mode = st.radio(
        "Tracking method",
        ["ByteTrack baseline", "Context-aware tracker"],
    )

    use_zones = st.checkbox(
        "Enable road-zone context",
        value=True,
        help="Applies two horizontal road zones based on video height.",
    )

    conf = st.slider(
        "Detection confidence",
        0.10,
        0.90,
        0.35,
        0.05,
    )

    imgsz = st.select_slider(
        "Model image size",
        options=[320, 416, 512, 640],
        value=640,
    )

    max_frames = st.number_input(
        "Maximum frames (0 = all)",
        min_value=0,
        max_value=10000,
        value=300,
        step=100,
    )

    st.divider()
    st.caption(
        "Context-aware matching combines predicted motion, bounding-box "
        "overlap, stable class history, zone consistency, and direction consistency."
    )


uploaded = st.file_uploader(
    "Upload road video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded:
    st.subheader("Input video")
    st.video(uploaded.getvalue())

    if uploaded.size > 200 * 1024 * 1024:
        st.error("Please upload a video smaller than 200 MB.")
        st.stop()

    if st.button(
        "Analyze video",
        type="primary",
        use_container_width=True,
    ):
        progress_bar = st.progress(0)
        status = st.empty()

        try:
            with tempfile.TemporaryDirectory(prefix="smartroad_") as td:
                suffix = Path(uploaded.name).suffix or ".mp4"

                input_path = os.path.join(td, "input" + suffix)
                output_path = os.path.join(td, "annotated.mp4")
                browser_path = os.path.join(td, "annotated_h264.mp4")
                csv_path = os.path.join(td, "tracking.csv")

                with open(input_path, "wb") as file:
                    file.write(uploaded.getvalue())

                import cv2

                cap = cv2.VideoCapture(input_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                zones = make_zones(width, height, use_zones)

                def progress(done, total, message):
                    progress_bar.progress(
                        min(1.0, done / max(total, 1))
                    )
                    status.write(message)

                result = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    csv_path=csv_path,
                    mode=mode,
                    conf=conf,
                    imgsz=int(imgsz),
                    max_frames=int(max_frames),
                    zones=zones,
                    progress=progress,
                )

                if not os.path.exists(output_path):
                    raise RuntimeError("Annotated video was not generated.")

                converted = convert_for_browser(
                    output_path,
                    browser_path,
                )

                final_video_path = browser_path if converted else output_path

                with open(final_video_path, "rb") as file:
                    video_bytes = file.read()

                with open(csv_path, "rb") as file:
                    csv_bytes = file.read()

            st.session_state["result"] = result
            st.session_state["video_bytes"] = video_bytes
            st.session_state["csv_bytes"] = csv_bytes
            st.session_state["selected_mode"] = mode
            st.session_state["zones_enabled"] = use_zones
            st.session_state["converted"] = converted

            progress_bar.progress(1.0)
            status.write("Analysis complete.")
            st.success("Video processed successfully.")

        except Exception as error:
            st.error(f"Analysis failed: {error}")

if "result" in st.session_state:
    r = st.session_state["result"]

    st.divider()
    st.header("Traffic analytics")

    a, b, c, d = st.columns(4)

    a.metric("Frames processed", f"{r['frames_processed']:,}")
    b.metric("Unique tracked objects", f"{r['unique_track_ids']:,}")
    c.metric("Object observations", f"{r['detection_records']:,}")
    d.metric("Processing speed", f"{r['fps']:.2f} FPS")

    st.caption(
        f"Method: {st.session_state['selected_mode']} | "
        f"Road zones: {'Enabled' if st.session_state['zones_enabled'] else 'Disabled'}"
    )

    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Annotated output video")
        st.video(st.session_state["video_bytes"])

        st.download_button(
            "Download annotated video",
            data=st.session_state["video_bytes"],
            file_name="smartroad_annotated.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    with right:
        st.subheader("Unique objects by class")

        unique_counts = r["unique_by_class"]

        if unique_counts:
            chart = pd.DataFrame(
                {"Unique objects": list(unique_counts.values())},
                index=list(unique_counts.keys()),
            )
            st.bar_chart(chart)
        else:
            st.info("No tracked objects found.")

        st.download_button(
            "Download tracking CSV",
            data=st.session_state["csv_bytes"],
            file_name="smartroad_tracking.csv",
            mime="text/csv",
            use_container_width=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Road-zone observations")

        zone_counts = r["observations_by_zone"]

        if zone_counts:
            zone_chart = pd.DataFrame(
                {"Observations": list(zone_counts.values())},
                index=list(zone_counts.keys()),
            )
            st.bar_chart(zone_chart)
        else:
            st.info("No zone data available.")

    with col2:
        st.subheader("Direction flow")

        direction_counts = r["direction_flow"]

        if direction_counts:
            direction_chart = pd.DataFrame(
                {"Observations": list(direction_counts.values())},
                index=list(direction_counts.keys()),
            )
            st.bar_chart(direction_chart)
        else:
            st.info(
                "Direction flow appears only in context-aware mode."
            )

    if r["records"]:
        st.subheader("Tracking records")

        st.dataframe(
            pd.DataFrame(r["records"]).head(1000),
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "This dashboard reports processing and traffic observations. "
        "To claim tracker accuracy improvement, compare both methods "
        "on manually annotated clips using ID switches, MOTA, IDF1, or HOTA."
    )

else:
    st.info("Upload a short road video, choose a method, and click Analyze video.")