import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import time

# Page configuration
st.set_page_config(
    page_title="Traffic Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .detection-box {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load the YOLO model with caching"""
    try:
        # Import torch to set weights_only parameter
        import torch
        
        # For PyTorch 2.6+, we need to handle the weights_only parameter
        # Set weights_only=False as the model is from a trusted source
        import warnings
        warnings.filterwarnings('ignore')
        
        # Load model with proper error handling
        model = YOLO(model_path)
        
        return model
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific guidance based on error type
        if "weights_only" in error_msg:
            st.error("⚠️ PyTorch Compatibility Issue Detected")
            st.warning("""
            **Solution Options:**
            
            1. **Quick Fix** - Run this command in terminal:
            ```bash
            export TORCH_LOAD_WEIGHTS_ONLY=0
            streamlit run app.py
            ```
            
            2. **Or** downgrade PyTorch:
            ```bash
            pip install torch==2.1.0 torchvision==0.16.0
            ```
            """)
        else:
            st.error(f"Error loading model: {error_msg}")
            st.info("💡 Make sure your `best.pt` file is in the same folder as this app.")
        
        return None

def process_image(image, model, conf_threshold):
    """Process a single image and return results"""
    # Convert PIL to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run inference
    results = model(image_cv, conf=conf_threshold)[0]
    
    # Custom plotting with smaller boxes and labels
    annotated_image = image_cv.copy()
    
    # Get image dimensions for scaling
    img_height, img_width = annotated_image.shape[:2]
    
    # Define colors for each class (BGR format for OpenCV)
    colors = {
        'bus': (76, 107, 255),      # Red
        'car': (196, 205, 78),      # Teal
        'motorbike': (109, 230, 255), # Yellow
        'truck': (211, 225, 149)    # Mint
    }
    
    # Calculate font scale and thickness based on image size
    font_scale = min(img_width, img_height) / 1000
    thickness = max(1, int(min(img_width, img_height) / 500))
    
    if len(results.boxes) > 0:
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class info
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            
            # Get color for this class
            color = colors.get(cls_name, (255, 107, 76))  # Default purple
            
            # Draw bounding box (thinner lines)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label = f"{cls_name} {conf:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw label background (smaller, above the box)
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            cv2.rectangle(
                annotated_image,
                (x1, label_y - text_height - baseline),
                (x1 + text_width, label_y + baseline),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text
                thickness,
                cv2.LINE_AA
            )
    
    # Convert back to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, results

def process_video(video_path, model, conf_threshold, progress_bar, status_text):
    """Process video and save output"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)[0]
        
        # Get annotated frame
        annotated_frame = results.plot()
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path

def get_detection_stats(results):
    """Extract detection statistics from results"""
    detections = {}
    
    if len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            
            if cls_name not in detections:
                detections[cls_name] = []
            detections[cls_name].append(conf)
    
    return detections

def main():
    # Header
    st.markdown('<p class="main-header">🚗 Traffic Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">YOLOv8-based Vehicle Detection | Real-time Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path input
        model_path = st.text_input(
            "Model Path",
            value="best.pt",
            help="Enter the path to your best.pt model file"
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        st.divider()
        
        # Upload type
        upload_type = st.radio(
            "Select Input Type",
            ["Image", "Video"],
            help="Choose whether to upload an image or video"
        )
        
        st.divider()
        
        # Info section
        st.info("""
        **Detected Classes:**
        - 🚌 Bus
        - 🚗 Car
        - 🏍️ Motorbike
        - 🚚 Truck
        """)
        
        st.success("""
        **Instructions:**
        1. Make sure `best.pt` is in the same folder
        2. Select input type (Image/Video)
        3. Upload your file
        4. Adjust confidence threshold
        5. Click 'Run Detection'
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload")
        
        if upload_type == "Image":
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a traffic image for detection"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
        else:  # Video
            uploaded_file = st.file_uploader(
                "Choose a video",
                type=['mp4', 'avi', 'mov'],
                help="Upload a traffic video for detection"
            )
            
            if uploaded_file is not None:
                # Save uploaded video temporarily
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video.write(uploaded_file.read())
                temp_video.close()
                
                st.video(uploaded_file)
    
    with col2:
        st.subheader("📊 Results")
        results_placeholder = st.empty()
    
    # Detection button
    if uploaded_file is not None:
        if st.button("🚀 Run Detection", type="primary"):
            # Check if model exists
            if not os.path.exists(model_path):
                st.error(f"❌ Model file '{model_path}' not found! Please make sure the model is in the correct location.")
                st.info("💡 Tip: Place your `best.pt` file in the same folder as this app.")
                return
            
            with st.spinner("🔄 Loading model..."):
                model = load_model(model_path)
            
            if model is None:
                return
            
            if upload_type == "Image":
                with st.spinner("🔍 Running detection..."):
                    start_time = time.time()
                    
                    # Process image
                    annotated_image, results = process_image(image, model, conf_threshold)
                    
                    # Get detection stats
                    detections = get_detection_stats(results)
                    
                    inference_time = time.time() - start_time
                
                with col2:
                    # Display results
                    st.image(annotated_image, caption="Detection Results", use_column_width=True)
                    
                    # Statistics
                    st.markdown("### 📈 Detection Statistics")
                    
                    # Metrics row
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Total Detections", len(results.boxes))
                    with metrics_col2:
                        st.metric("Inference Time", f"{inference_time:.2f}s")
                    with metrics_col3:
                        st.metric("Unique Classes", len(detections))
                    
                    # Detailed detections
                    if detections:
                        st.markdown("### 🎯 Detected Objects")
                        for cls_name, confidences in detections.items():
                            avg_conf = np.mean(confidences)
                            count = len(confidences)
                            
                            # Color scheme for different classes
                            color_map = {
                                'bus': '#FF6B6B',      # Red
                                'car': '#4ECDC4',      # Teal
                                'motorbike': '#FFE66D', # Yellow
                                'truck': '#95E1D3',    # Mint
                            }

                            # Get color for this class, default to blue
                            box_color = color_map.get(cls_name.lower(), '#6C5CE7')

                            st.markdown(
                                f"""
                                <div class="detection-box" style="background-color: {box_color}; color: #000000; font-weight: 500;">
                                    <strong style="font-size: 1.1em;">🚦 {cls_name.upper()}</strong>: {count} detected | 
                                    Avg. Confidence: {avg_conf:.2%}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.warning("No objects detected. Try lowering the confidence threshold.")
            
            else:  # Video processing
                with col2:
                    st.markdown("### 🎬 Processing Video...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Process video
                output_path = process_video(
                    temp_video.name,
                    model,
                    conf_threshold,
                    progress_bar,
                    status_text
                )
                
                with col2:
                    status_text.text("✅ Processing complete!")
                    st.success("Video processing completed successfully!")
                    
                    # Display processed video
                    with open(output_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f,
                            file_name="detected_traffic.mp4",
                            mime="video/mp4"
                        )
                
                # Cleanup
                os.unlink(temp_video.name)
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p> <strong>Traffic Detection System</strong> | Powered by YOLOv8</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
