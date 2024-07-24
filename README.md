### AI House Abuse Sign Detection using Python and MediaPipe

**Overview:**
This project utilizes Python and MediaPipe library to detect potential signs of abuse using hand gestures. It processes real-time video input from a webcam, analyzes hand movements, and flags instances where specific gestures indicative of abuse are detected.

**How it Works:**
1. **Hand Gesture Detection:** The program uses the MediaPipe Hands module to detect and track hand landmarks in the video feed. These landmarks represent key points on the hand such as fingertips, knuckles, and the palm.
  
2. **Gesture Analysis:**
   - **Finger Curl Detection:** Monitors the curling of fingers (index, middle, ring) to assess if they are fully extended or not.
   - **Thumb Position Detection:** Checks the position of the thumb to determine if it is curled inward or outward.
   - **Abuse Sign Detection:** Combines the analysis of finger curls and thumb position to detect potential signs of abuse. For example, if all fingers are curled and the thumb is not extended, it may indicate a signal for help.

3. **Real-time Feedback:** The program provides real-time visual feedback by drawing hand landmarks, angles between finger joints, and displaying textual indications of gesture states (e.g., fingers curled, thumb curled).

4. **Dependencies:**
   - **Python Libraries:** OpenCV (`cv2`), NumPy (`numpy`), MediaPipe (`mediapipe`).
   - **Hardware:** Requires a webcam or video input device to capture real-time video.

**Usage:**
- **Setup:** Ensure Python is installed along with necessary libraries (`cv2`, `numpy`, `mediapipe`).
- **Execution:** Run the script, which initializes the webcam, processes the video feed, and displays the output in a graphical window.
- **Interpretation:** Users can observe the real-time analysis of hand gestures and interpret if signs of abuse are detected based on the displayed feedback.

**Conclusion:**
This project demonstrates a practical application of AI in identifying potential signs of abuse using accessible hardware and software tools. It aims to raise awareness and potentially aid in the detection of distress signals through hand gestures, leveraging computer vision techniques for social benefit.
