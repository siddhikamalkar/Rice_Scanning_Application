Rice Scanning Application (AI/ML)

Project Overview
----------------
This project is a machine learning–based rice scanning application that analyzes rice images to detect:
- Total number of rice grains
- Number of broken grains
- Overall model accuracy (approximately 70–80%)

The application uses OpenCV for image preprocessing and contour-based grain detection, and a Support Vector Machine (SVM) classifier to distinguish broken and whole rice grains based on geometric features.

Technology Stack
----------------
- Python
- OpenCV (image preprocessing & contour detection)
- Scikit-learn (Support Vector Machine)
- Flask (backend)
- HTML, CSS, minimal JavaScript (frontend)
Dataset Usage
-------------
Images containing only broken grains and only whole grains were used for model training to ensure clean and correct labels at the grain level. Images containing mixed grains were excluded from training and used only during inference and final testing to simulate real-world usage.


Workflow Summary
----------------
1. Rice images are preprocessed using grayscale conversion, Gaussian blur, thresholding, and morphological operations.
2. Contour detection is applied, where each contour represents a single rice grain.
3. Features such as area, height, and aspect ratio are extracted for each grain.
4. An SVM model is trained using labeled grain-level features (broken vs whole).
5. The trained model achieves approximately 74.6% accuracy on unseen test data.
6. A Flask web application allows users to upload rice images and view analysis results.

Model Performance
-----------------
Model Used: Support Vector Machine (SVM)
Accuracy Achieved: ~74.6%

Note: Accuracy is calculated during model training using a held-out test dataset. Image-level accuracy is not computed due to the absence of ground truth labels for uploaded images.

How to Run the Project
----------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train_svm.py

3. Run the application:
   python app.py

4. Open in browser:
   http://127.0.0.1:5000

Model Files
-----------
Trained model files (.pkl) are not included in the repository due to GitHub file size limitations.
The model can be generated locally by running the training script provided.

Assumptions & Limitations
-------------------------
- Input images are assumed to contain rice grains.
- Mixed-grain images are used only during inference.
- Image quality and lighting conditions may affect performance.

Author
------
Siddhi

