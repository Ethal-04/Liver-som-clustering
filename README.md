# Self-Organizing Map for Liver Blood Test Clustering

This project applies a Self-Organizing Map (SOM) to cluster and visualize liver function blood test data.  
The goal is to identify natural groupings among patients based on key biochemical parameters without supervised labels.



## 📊 Dataset
- **Source**: Indian Liver Patient Dataset (ILPD) from the UCI Machine Learning Repository.
- **Features Used**:
  - Total Bilirubin
  - Direct Bilirubin
  - Alkaline Phosphatase
  - Aspartate Aminotransferase (AST)
  - Albumin



## ⚙️ Project Structure
- `liver_som_clustering.py`: Main Python script implementing SOM training, visualization, and patient mapping.
- `liverdata.txt`: Preprocessed dataset used for clustering.
- `README.md`: Project description and instructions.



## 🚀 How to Run
1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn minisom
   ```
2. Run the main script:
  ```bash
  python liver_som_clustering.py
  ```
3. Enter a new patient's blood parameters (Total Bilirubin, Direct Bilirubin, Alkaline Phosphatase, AST, Albumin) when prompted to map them on the SOM.


🧠 Methodology


Data normalization using Min-Max scaling.

SOM grid size: 7x7 neurons.

Training iterations: 200.

Visualize patient clusters and identify healthy vs diseased profiles based on clustering.


🏥 Clinical Relevance


This project demonstrates how unsupervised machine learning (SOM) can aid early detection of liver dysfunction by clustering blood test patterns, enabling faster clinical decisions without requiring extensive labeled data.

📜 License

This project is licensed under the MIT License.
If you use this work or build upon it, please consider citing the relevant references included in the project.
