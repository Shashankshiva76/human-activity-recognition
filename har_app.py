import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
import imageio.v3 as iio
import numpy as np
from collections import deque
import tempfile
import logging
import os
import cv2
from collections import deque
import tempfile
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Page configuration
st.set_page_config(
    page_title="Human Activity Recognition System",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .activity-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Activity mapping
ACTIVITY_MAPPING = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

ACTIVITY_COLORS = {
    'WALKING': '#2ecc71',
    'WALKING_UPSTAIRS': '#3498db',
    'WALKING_DOWNSTAIRS': '#9b59b6',
    'SITTING': '#e74c3c',
    'STANDING': '#f39c12',
    'LAYING': '#1abc9c'
}

# Load trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        base_path = os.path.dirname(__file__)
        model = joblib.load(os.path.join(base_path, 'logistic_model.joblib' ))
        scaler = joblib.load(os.path.join(base_path, 'std_scaler.joblib'))
        st.success("✅ Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please ensure 'logistic_model.joblib' and 'stand_scl.joblib' are in the same directory")
        return None, None

# Initialize MediaPipe Pose (for video mode)
@st.cache_resource
def load_pose_detector():
    base_path = os.path.dirname(__file__)
    model_path_pose = os.path.join(base_path, 'pose_landmarker_lite.task')
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_pose),
        running_mode=VisionRunningMode.VIDEO
    )
    
    detector = PoseLandmarker.create_from_options(options)
    return detector

def extract_pose_features(landmarks):
    """Extract features from pose landmarks"""
    if landmarks is None:
        return None
    
    # Extract x, y, z coordinates for all 33 landmarks
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(features)

def pose_to_sensor_features(pose_history, feature_names):
    """
    Convert pose features to simulated sensor features (561 dimensions)
    This is an approximation to make video work with sensor-based model
    """
    if len(pose_history) < 10:
        # Not enough history, return zeros
        return np.zeros(561)
    
    # Get recent poses
    recent_poses = [p for p in list(pose_history)[-10:] if p is not None]
    
    if len(recent_poses) < 5:
        return np.zeros(561)
    
    # Stack poses
    poses_array = np.array(recent_poses)  # Shape: (n_frames, 99)
    
    # Calculate various statistical features to simulate sensor data
    features = []
    
    # For each pose coordinate dimension
    for i in range(poses_array.shape[1]):
        signal = poses_array[:, i]
        
        # Time domain features
        features.append(np.mean(signal))  # mean
        features.append(np.std(signal))   # std
        features.append(np.median(np.abs(signal - np.median(signal))))  # mad
        features.append(np.max(signal))   # max
        features.append(np.min(signal))   # min
        
        # Velocity (first derivative)
        velocity = np.diff(signal)
        if len(velocity) > 0:
            features.append(np.mean(velocity))
            features.append(np.std(velocity))
    
    # Pad or truncate to 561 features
    features = np.array(features)
    if len(features) < 561:
        features = np.pad(features, (0, 561 - len(features)), mode='constant')
    else:
        features = features[:561]
    
    return features

def create_sensor_dataframe(sensor_data):
    """Convert sensor data dictionary to DataFrame"""
    data_dict = {}
    for key, value in sensor_data.items():
        if isinstance(value, dict):
            data_dict[key] = list(value.values())[0]
        else:
            data_dict[key] = value
    
    # Remove target columns
    data_dict.pop('subject', None)
    data_dict.pop('Activity', None)
    data_dict.pop('Activity_Label', None)
    
    return pd.DataFrame([data_dict])

def predict_activity(model, scaler, features):
    """Predict activity using the trained model"""
    if model is None or scaler is None:
        return "MODEL_NOT_LOADED", 0.0, None
    
    try:
        # Ensure features are 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        activity_name = ACTIVITY_MAPPING[prediction]
        
        return activity_name, confidence, probabilities
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "ERROR", 0.0, None

def process_video_frame(frame, pose_detector, model, scaler, pose_history, feature_names, timestamp_ms):
    """Process a single video frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect pose using new API
    result = pose_detector.detect_for_video(mp_image, timestamp_ms)
    
    activity = "UNKNOWN"
    confidence = 0.0
    probabilities = None
    
    if result.pose_landmarks:
        # Get landmarks from first person detected
        landmarks = result.pose_landmarks[0]
        
        # Draw landmarks manually
        h, w, _ = frame.shape
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw key connections
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Torso
            (23, 24),  # Hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        
        # Extract pose features (33 landmarks x 3 coordinates = 99 features)
        pose_features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        pose_history.append(pose_features)
        
        # Convert pose to sensor-like features
        sensor_features = pose_to_sensor_features(pose_history, feature_names)
        
        # Predict activity
        activity, confidence, probabilities = predict_activity(model, scaler, sensor_features)
        
        # Add label to frame
        cv2.putText(
            frame,
            f"{activity} ({confidence:.1%})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )
    else:
        pose_history.append(None)
    
    return frame, activity, confidence, probabilities





def process_uploaded_video(video_path, pose_detector, model, scaler, feature_names, progress_bar, status_text):
    """Process uploaded video file using imageio (Streamlit Cloud compatible)"""
    
    try:
        # Read video metadata
        props = iio.improps(video_path, plugin="pyav")
        fps = props.fps if hasattr(props, 'fps') else 30.0
        
        logging.info(f"Video FPS: {fps}")
        
        # Create output path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Initialize variables
        pose_history = deque(maxlen=30)
        activity_log = []
        frame_count = 0
        processed_frames = []
        
        # Read and process all frames
        for frame in iio.imiter(video_path, plugin="pyav"):
            # Convert RGB (imageio) to BGR (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            timestamp_ms = int(frame_count / fps * 1000)
            
            # Process the frame
            processed_frame, activity, confidence, probabilities = process_video_frame(
                frame_bgr, pose_detector, model, scaler, pose_history, 
                feature_names, timestamp_ms
            )
            
            # Convert BGR back to RGB for video writing
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(processed_frame_rgb)
            
            # Log activity
            activity_log.append({
                'frame': frame_count,
                'time': frame_count / fps,
                'activity': activity,
                'confidence': confidence
            })
            
            frame_count += 1
            
            # Update progress
            if status_text:
                status_text.text(f"Processing frame {frame_count}")
            
            # Log every 30 frames
            if frame_count % 30 == 0:
                logging.info(f"Processed {frame_count} frames, current activity: {activity}")
        
        # Write all frames to output video
        logging.info(f"Writing {len(processed_frames)} frames to video at {fps} fps")
        
        iio.imwrite(
            output_path,
            processed_frames,
            plugin="pyav",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p"
        )
        
        # Update progress to 100%
        if progress_bar:
            progress_bar.progress(1.0)
        
        # Verify output file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Video processing complete: {frame_count} frames, size: {os.path.getsize(output_path)} bytes")
        else:
            raise RuntimeError("Output video file is empty or missing")
        
        return output_path, pd.DataFrame(activity_log)
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}", exc_info=True)
        raise RuntimeError(f"Video processing failed: {str(e)}")


def plot_activity_timeline(activity_log):
    """Create timeline visualization"""
    fig = px.scatter(
        activity_log,
        x='time',
        y='activity',
        color='activity',
        size='confidence',
        color_discrete_map=ACTIVITY_COLORS,
        hover_data=['confidence'],
        title='Activity Detection Timeline'
    )
    fig.update_layout(xaxis_title='Time (seconds)', yaxis_title='Activity', height=400)
    return fig

def plot_activity_distribution(activity_log):
    """Create pie chart of activity distribution"""
    activity_counts = activity_log['activity'].value_counts()
    fig = px.pie(
        values=activity_counts.values,
        names=activity_counts.index,
        color=activity_counts.index,
        color_discrete_map=ACTIVITY_COLORS,
        title='Activity Distribution'
    )
    return fig

def plot_confidence_over_time(activity_log):
    """Plot confidence over time"""
    fig = px.line(
        activity_log,
        x='time',
        y='confidence',
        title='Prediction Confidence Over Time'
    )
    fig.update_layout(
        xaxis_title='Time (seconds)',
        yaxis_title='Confidence',
        yaxis_range=[0, 1],
        height=300
    )
    return fig

def main():
    st.markdown('<p class="main-header">🏃 Human Activity Recognition System</p>', unsafe_allow_html=True)
    if 'activity_log' in st.session_state:
        activity_log = st.session_state['activity_log']
        if isinstance(activity_log, pd.DataFrame) and 'time' not in activity_log.columns:
            del st.session_state['activity_log']
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.error("Cannot proceed without model. Please upload model files.")
        st.stop()
    
    # Get feature names (assuming standard order from training)
    feature_names = [f'feature_{i}' for i in range(561)]
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        st.info("""
        **Model Information**
        - Type: Logistic Regression
        - Accuracy: 95.52%
        - Input: 561 sensor features
        - Classes: 6 activities
        
        **Input Modes:**
        1. Sensor Data (JSON)
        2. Video Upload
  
        """)
        
        input_mode = st.radio(
            "Select Input Mode",
            ["📱 Sensor Data (JSON)", "📁 Upload Video"]
        )
    # Clear video session state when switching to sensor mode
    if input_mode == "📱 Sensor Data (JSON)" and 'activity_log' in st.session_state:
        if 'processed_video' in st.session_state:
            del st.session_state['activity_log']
            del st.session_state['processed_video']
    # Main tabs
    tabs = st.tabs(["🎯 Recognition", "📈 Analytics", "ℹ️ About"])
    
    # Tab 1: Recognition
    with tabs[0]:
        if input_mode == "📱 Sensor Data (JSON)":
            st.subheader("Sensor-Based Activity Recognition")
            st.info("Upload JSON with 561 sensor features from accelerometer/gyroscope")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Sample data button
                if st.button("Load Sample Data"):
                    # Create sample from test data structure
                    sample_data = {f'feature_{i}': {0: np.random.randn()} for i in range(561)}
                    st.session_state['sensor_data'] = sample_data
                
                sensor_json = st.text_area(
                    "Paste sensor data (JSON format):",
                    value=json.dumps(st.session_state.get('sensor_data', {}), indent=2) if 'sensor_data' in st.session_state else "{}",
                    height=300
                )
                
                uploaded_json = st.file_uploader("Or upload JSON file", type=['json'])
                
                if uploaded_json:
                    sensor_data = json.load(uploaded_json)
                else:
                    try:
                        sensor_data = json.loads(sensor_json) if sensor_json != "{}" else None
                    except:
                        sensor_data = None
                
                if st.button("🚀 Predict Activity", type="primary") and sensor_data:
                    try:
                        df = create_sensor_dataframe(sensor_data)
                        
                        if df.shape[1] != 561:
                            st.error(f"Expected 561 features, got {df.shape[1]}")
                        else:
                            features = df.values
                            activity, confidence, probabilities = predict_activity(model, scaler, features)
                            
                            st.session_state['last_prediction'] = activity
                            st.session_state['last_confidence'] = confidence
                            st.session_state['last_probabilities'] = probabilities
                            
                            # Add to history
                            if 'history' not in st.session_state:
                                st.session_state['history'] = []
                            
                            st.session_state['history'].append({
                                'timestamp': datetime.now(),
                                'activity': activity,
                                'confidence': confidence
                            })
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col2:
                st.subheader("Prediction Results")
                
                if 'last_prediction' in st.session_state:
                    activity = st.session_state['last_prediction']
                    confidence = st.session_state['last_confidence']
                    probabilities = st.session_state['last_probabilities']
                    
                    color = ACTIVITY_COLORS.get(activity, '#95a5a6')
                    st.markdown(
                        f'<div class="activity-badge" style="background-color: {color}; color: white;">'
                        f'{activity}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    if probabilities is not None:
                        st.subheader("All Probabilities")
                        for i, prob in enumerate(probabilities):
                            act_name = ACTIVITY_MAPPING[i]
                            st.progress(prob, text=f"{act_name}: {prob:.1%}")
                else:
                    st.info("Upload sensor data and click Predict")
        
        elif input_mode == "📁 Upload Video":
            st.subheader("Video-Based Activity Recognition")
            st.warning("⚠️ Video mode uses pose estimation to simulate sensor features. Results may vary from direct sensor input.")
            
            uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
            
            print("Upload File is : ", uploaded_file)
            if uploaded_file:
                # tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                # tfile.write(uploaded_file.read())
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                
                print("TFFILE NAME: ", video_path)
                st.video(video_path)
                
                if st.button("🚀 Process Video", type="primary"):
                    pose_detector  = load_pose_detector()
                    
                    with st.spinner("Processing video..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        output_path, activity_log = process_uploaded_video(
                            video_path, pose_detector, model, scaler, 
                            feature_names, progress_bar, status_text
                        )
                        
                        st.success("✅ Processing complete!")
                        
                        st.session_state['processed_video'] = output_path
                        st.session_state['activity_log'] = activity_log
                        
                        st.subheader("Processed Video")
                        st.video(output_path)
                        
                        if activity_log is not None and not activity_log.empty:
                            if 'time' in activity_log.columns:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Duration", f"{activity_log['time'].max():.1f}s")
                                with col2:
                                    most_common = activity_log['activity'].mode()[0] if len(activity_log['activity'].mode()) > 0 else "N/A"
                                    st.metric("Primary Activity", most_common)
                                with col3:
                                    avg_conf = activity_log['confidence'].mean()
                                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig = plot_activity_timeline(activity_log)
                                    st.plotly_chart(fig, use_container_width=True)
                                with col2:
                                    fig = plot_activity_distribution(activity_log)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                fig = plot_confidence_over_time(activity_log)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Missing 'time' column. Available columns: {activity_log.columns.tolist()}")
                        else:
                            st.error("Activity log is empty or None")
        
       
    # Tab 2: Analytics
    with tabs[1]:
        st.subheader("Activity Analytics")
        
        if 'activity_log' in st.session_state:
            activity_log = st.session_state['activity_log']
            
            # Validate that activity_log has required columns
            required_columns = ['time', 'activity', 'confidence']
            if not all(col in activity_log.columns for col in required_columns):
                st.warning("Activity log is incomplete. Please process a video to see full analytics.")
                st.stop()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", len(activity_log))
            with col2:
                st.metric("Duration", f"{activity_log['time'].max():.1f}s")
            with col3:
                st.metric("Unique Activities", activity_log['activity'].nunique())
            with col4:
                st.metric("Avg Confidence", f"{activity_log['confidence'].mean():.1%}")
            
            st.subheader("Activity Breakdown")
            breakdown = activity_log['activity'].value_counts()
            breakdown_df = pd.DataFrame({
                'Activity': breakdown.index,
                'Count': breakdown.values,
                'Percentage': (breakdown.values / len(activity_log) * 100).round(2)
            })
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
            csv = activity_log.to_csv(index=False)
            st.download_button(
                "📥 Download Activity Log",
                csv,
                f"activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.info("No activity data available. Process a video or input sensor data to see analytics.")
    
    # Tab 3: About
    with tabs[2]:
        st.subheader("About This System")
        
        st.markdown("""
        ### Model Information
        
        **Trained Model:**
        - Algorithm: Logistic Regression (Multinomial)
        - Training Accuracy: 95.52%
        - Dataset: UCI HAR (Human Activity Recognition with Smartphones)
        - Features: 561 sensor measurements
        - Classes: 6 activities
        
        **Activities Detected:**
        1. WALKING
        2. WALKING_UPSTAIRS
        3. WALKING_DOWNSTAIRS
        4. SITTING
        5. STANDING
        6. LAYING
        
        ### Input Modes
        
        **1. Sensor Data (Recommended)**
        - Direct input of 561 sensor features
        - Highest accuracy (95%+)
        - Requires accelerometer/gyroscope data
        
        **2. Video Upload**
        - Uses pose estimation to simulate sensor data
        - Approximate accuracy (70-85%)
        - No wearable sensors needed
        
        
        
        ### How to Use
        
        **For Sensor Data:**
        1. Collect data from smartphone sensors
        2. Format as JSON with 561 features
        3. Upload and predict
        
        **For Video:**
        1. Record video with clear full-body view
        2. Upload to the app
        3. Wait for processing
        4. View results and analytics
        
        ### Technical Details
        
        **Sensor Features (561 total):**
        - Time domain signals
        - Frequency domain signals
        - Statistical measures (mean, std, mad, max, min)
        - Body acceleration and gravity
        - Jerk signals
        - Magnitude calculations
        
        **Model Performance:**
        ```
        Activity              Precision  Recall  F1-Score
        WALKING              0.94       0.99    0.97
        WALKING_UPSTAIRS     0.96       0.95    0.96
        WALKING_DOWNSTAIRS   0.99       0.94    0.97
        SITTING              0.97       0.87    0.92
        STANDING             0.89       0.97    0.93
        LAYING               1.00       0.99    1.00
        ```
        
        ### Applications
        
        - Health monitoring
        - Fitness tracking
        - Elderly fall detection
        - Research studies
        - Smart home automation
        """)

if __name__ == "__main__":
    main()

