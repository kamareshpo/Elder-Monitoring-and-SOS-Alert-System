# 🧠 AI-Based Elder Monitoring System

## 📌 Overview
The **AI-Based Elder Monitoring System** is a real-time safety solution designed to monitor elderly individuals using computer vision and AI techniques.  
It detects **falls, emergency gestures, and inactivity**, and sends instant alerts with media evidence to caregivers.

---

## 🎯 Features

- 🚨 **Fall Detection**
  - Multi-condition scoring (pose, velocity, body angle)
  - Time-based validation to reduce false alerts

- ✋ **SOS Gesture Detection**
  - Both hands raised → emergency alert  
  - Hand on chest (held for 5 seconds) → SOS trigger  
  - Stability filtering to avoid random detections  

- 💤 **No Movement Detection**
  - Detects inactivity for long duration  
  - Sends alert if no movement is detected  

- 📹 **Auto Video Recording**
  - Records video + audio during alert  
  - Captures snapshot for quick verification  

- 🌙 **Night Vision Mode**
  - Enhanced visibility in low-light conditions  

- 🔔 **Alert System**
  - Telegram notifications (photo + video + message)  
  - Local buzzer alert  
  - Mobile alarm trigger  

---

## 🛠 Tech Stack

- **Programming:** Python  
- **Computer Vision:** OpenCV, MediaPipe  
- **Backend:** Flask  
- **APIs:** Telegram Bot API  
- **Audio/Video:** FFmpeg, SoundDevice  
- **Others:** NumPy, Threading  

---
