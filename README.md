# 🚗 Traffic Detection System - Setup Guide

## Project Information
**Course:** Artificial Intelligence  AI-2002

**Project:** YOLOv8-based Traffic Detection System for Local Traffic Dataset

---

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ Python 3.8-3.10 installed
- ✅ `best.pt` model file 
- ✅ Sample images or videos for testing

---

## 🚀 Step-by-Step Setup Instructions

### **Step 1: Organize Your Files**

Create a project folder and organize it like this:

```
traffic-detection-app/
│
├── app.py                 # Streamlit app (provided)
├── requirements.txt       # Dependencies (provided)
├── best.pt               # Your trained model (from Phase 1)
├── README.md             # This file
└── test_data/            # Optional: folder for test images/videos
    ├── sample1.jpg
    ├── sample2.jpg
    └── traffic_video.mp4
```

**Important:** Make sure your `best.pt` file is in the same folder as `app.py`!

---

### **Step 2: Create a Virtual Environment (Recommended)**

Open your terminal/command prompt and navigate to your project folder:

```bash
# Navigate to your project folder
cd path/to/traffic-detection-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

---

### **Step 3: Install Dependencies**

With the virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web framework)
- Ultralytics (YOLOv8)
- OpenCV (image/video processing)
- PIL (image handling)
- NumPy (array operations)
- PyTorch (deep learning backend)

**Note:** Installation may take 5-10 minutes depending on your internet speed.

---

### **Step 4: Verify Your Model File**

Make sure your `best.pt` file is in the correct location:

```bash
# On Windows:
dir

# On Mac/Linux:
ls -la
```

You should see `best.pt` listed in the output.

---

### **Step 5: Run the Application**


**Windows:**
```bash
streamlit run app.py
```

**Mac/Linux:**
```bash
export TORCH_LOAD_WEIGHTS_ONLY=0
streamlit run app.py
```

The terminal will show:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

### **Step 6: Use the Application**

1. **Open your browser** - The app should open automatically at `http://localhost:8501`

2. **Check the sidebar** - You'll see:
   - Model Path (should show "best.pt")
   - Confidence Threshold slider
   - Input type selector (Image/Video)

3. **Upload a file**:
   - Click "Browse files" 
   - Select an image (.jpg, .png) or video (.mp4, .avi)
   - Preview will appear

4. **Adjust settings**:
   - Move the confidence slider (default: 0.25)
   - Lower = more detections (but more false positives)
   - Higher = fewer detections (but more accurate)

5. **Run Detection**:
   - Click "🚀 Run Detection" button
   - Wait for processing
   - View results on the right side

6. **Analyze Results**:
   - See annotated image/video with bounding boxes
   - Check detection statistics
   - View class counts and confidence scores

7. **For videos**:
   - Processing may take time depending on video length
   - Watch the progress bar
   - Download processed video when done

---

## 🎯 Features

### ✨ What the App Can Do:

1. **Image Detection**
   - Upload traffic images
   - Real-time detection with bounding boxes
   - Shows detected classes (Bus, Car, Motorbike, Truck)
   - Displays confidence scores
   - Shows detection statistics

2. **Video Detection**
   - Process entire videos frame-by-frame
   - Progress tracking
   - Download processed video
   - Real-time preview

3. **Interactive Controls**
   - Adjustable confidence threshold
   - Easy file upload
   - Clean, professional interface

4. **Detection Analytics**
   - Total object count
   - Inference time
   - Unique class count
   - Per-class statistics
   - Average confidence per class

---

## 🛠️ Troubleshooting

### Problem: "Model file 'best.pt' not found"
**Solution:** 
- Make sure `best.pt` is in the same folder as `app.py`
- Check the spelling (case-sensitive on Mac/Linux)
- Try absolute path in the sidebar input box

### Problem: "No module named 'streamlit'"
**Solution:**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

### Problem: "CUDA out of memory" (if using GPU)
**Solution:**
- Reduce video resolution
- Process fewer frames
- Use CPU instead (model will auto-detect)

### Problem: App runs but shows errors
**Solution:**
- Check Python version: `python --version` (need 3.8+)
- Update pip: `pip install --upgrade pip`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Problem: Video processing is very slow
**Solution:**
- This is normal for CPU processing
- Use shorter videos for demo
- Consider using Google Colab with GPU for faster processing

### Problem: Upload button not working
**Solution:**
- Check file format (only .jpg, .png for images; .mp4, .avi for videos)
- Try smaller file size (< 200MB)
- Restart the app

---

## 💡 Tips for Best Results

1. **Image Quality**
   - Use clear, well-lit images
   - Avoid blurry or low-resolution images
   - Front/side views work best

2. **Confidence Threshold**
   - Start with 0.25
   - Increase if too many false detections
   - Decrease if missing obvious vehicles

3. **Video Processing**
   - Use short clips (10-30 seconds) for demos
   - Ensure good lighting in videos
   - Stable camera angle works best

4. **Testing**
   - Test with various scenarios:
     - Day/night traffic
     - Different weather conditions
     - Various vehicle densities
     - Different angles

---

## 📊 Expected Output

### For Images:
- Bounding boxes around detected vehicles
- Class labels (Bus, Car, Motorbike, Truck)
- Confidence scores per detection
- Summary statistics table

### For Videos:
- Processed video with detections on every frame
- Downloadable output file
- Processing time information

---

## 🎥 Demo Workflow

### Quick Demo Steps:

1. **Prepare Test Data**
   - Get 2-3 traffic images
   - Get 1 short video clip (10-20 seconds)

2. **Start App**
   ```bash
   streamlit run app.py
   ```

3. **Test Image Detection**
   - Select "Image" in sidebar
   - Upload first image
   - Click "Run Detection"
   - Show results
   - Try different confidence values

4. **Test Video Detection**
   - Select "Video" in sidebar
   - Upload video
   - Click "Run Detection"
   - Wait for processing
   - Download and show result

5. **Demonstrate Features**
   - Change confidence threshold
   - Compare different images
   - Show statistics panel
   - Explain class detections

---

## 📦 Sharing Your Project

### For Submission:
1. Create a ZIP file with:
   - `app.py`
   - `requirements.txt`
   - `best.pt`
   - `README.md`
   - Sample test images/videos

2. Include screenshots of:
   - App interface
   - Detection results
   - Statistics panel

### For GitHub:
```bash
git init
git add app.py requirements.txt README.md
git commit -m "Add traffic detection Streamlit app"
git remote add origin YOUR_REPO_URL
git push -u origin main
```

**Note:** Don't upload large model files to GitHub. Use Git LFS or provide download link.

---

## 🔄 Updating the App

If you want to modify the app:

1. Edit `app.py` with your changes
2. Save the file
3. Streamlit will auto-detect changes and ask to rerun
4. Click "Rerun" or press 'R' in the terminal

---

## 📞 Support

If you encounter issues:

1. Check this README first
2. Verify all files are in correct locations
3. Ensure Python version is 3.8+
4. Make sure all dependencies are installed
5. Check terminal output for specific error messages

---

## 🎓 Project Context

This application is the **Phase 2** deliverable for the AI course project. It demonstrates:

- Practical deployment of trained YOLOv8 model
- Real-world application development
- User-friendly interface design
- Integration of ML model with web framework
- Professional software presentation

**Phase 1:** Model training on local traffic dataset  
**Phase 2:** Frontend application for practical use cases

---

## 📝 License & Credits

**Model:** YOLOv8 by Ultralytics  
**Framework:** Streamlit  
**Dataset:** Local Traffic Dataset  



## ✅ Quick Reference Commands

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app 

# Windows:
 run app.py

# Mac/Linux:
export TORCH_LOAD_WEIGHTS_ONLY=0 && streamlit run app.py

# Deactivate virtual environment
deactivate
```

---

## 🎉 You're All Set!

Your traffic detection system is ready to use. Good luck with your demo! 🚀
