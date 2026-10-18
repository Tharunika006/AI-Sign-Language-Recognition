\# AI Sign Language Recognition Using Python and KNN



A real-time AI-based Sign Language Recognition system developed using Python, MediaPipe, OpenCV, and the K-Nearest Neighbors (KNN) machine learning algorithm.



The system uses a webcam to detect hand landmarks, recognize selected sign gestures, display the predicted signs, and provide speech output.



\## Project Overview



Communication can be challenging for people who use sign language when interacting with people unfamiliar with it.



This project aims to help bridge that communication gap by recognizing selected hand gestures through a webcam and converting recognized signs into text and speech.



The system uses MediaPipe to extract hand landmarks and a trained KNN classifier to predict the corresponding sign.



\## Features



\- Real-time hand detection using a webcam.

\- Hand landmark extraction using MediaPipe.

\- Sign classification using the KNN algorithm.

\- Recognition of five custom sign classes.

\- Displays recognized signs on the screen.

\- Supports sentence formation from recognized signs.

\- Text-to-speech output using Python speech libraries.

\- Custom dataset collection and model training.

\- Model evaluation and parameter tuning.



\## Supported Sign Gestures



The current model supports these five classes:



| Sign | Label |

|---|---|

| Hello | HELLO |

| No | NO |

| Stop | STOP |

| Thank You | THANK YOU |

| Yes | YES |



\*\*Note:\*\* This is a prototype that recognizes five trained gesture classes. It does not recognize every sign in a complete sign language.



\## Technologies Used



\- \*\*Programming Language:\*\* Python

\- \*\*Computer Vision:\*\* OpenCV

\- \*\*Hand Tracking:\*\* MediaPipe

\- \*\*Machine Learning:\*\* Scikit-learn KNN classifier

\- \*\*Speech Output:\*\* pyttsx3

\- \*\*Data Processing:\*\* NumPy and Pandas

\- \*\*Development Environment:\*\* Windows and Command Prompt



\## Project Structure



```text

AI-Sign-Language-Recognition/

│

├── dataset/

│   ├── HELLO.csv

│   ├── NO.csv

│   ├── STOP.csv

│   ├── THANK YOU.csv

│   └── YES.csv

│

├── models/

│   └── hand\_landmarker.task

│

├── collect\_data.py

├── compare\_models.py

├── evaluate\_model.py

├── predict\_knn.py

├── train\_knn.py

├── tune\_knn.py

├── sign\_language\_knn.pkl

└── .gitignore

