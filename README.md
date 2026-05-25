# Intelligent-Clinical-Early-Warning-System

**Deep Learning
Complex Computing Problem 
 “Can AI Save Lives?**
 
# **Building an Intelligent Clinical Early Warning System”**

THE CHALLENGE

Every year, thousands of patients in hospitals deteriorate silently, their vitals shift gradually, their clinical 
notes grow more urgent, and yet the warning signs go unnoticed until it is too late. Hospitals are now 
turning to AI to build Early Warning Systems that monitor patients continuously and flag those at risk of 
sepsis, cardiac arrest, or ICU transfer before a crisis occurs.

You have been hired as a Machine Learning Engineer at a healthcare AI startup. Your team has been 
given access to a real-world clinical dataset and tasked with building a deep learning pipeline that can 
predict patient deterioration risk from a combination of time-series vital signs and free-text clinical notes. 

Your models must be accurate but more importantly, your decisions must be explainable, because in 
healthcare a doctor needs to understand why the model flagged a patient, not just that it did.
Dataset: Use the PhysioNet Sepsis Prediction Challenge dataset (publicly available), or the Patient 
Survival Prediction dataset on Kaggle. If access permits, the MIMIC-III Clinical Notes dataset is also 
acceptable.

You will construct three generations of models, each solving a limitation of the previous one, mirroring 
how the field of deep learning actually evolved.

 PART A — IMPLEMENTATION 5 Marks

**Generation 1: Establishing the Baseline** 

"Before we trust AI with patient lives, we need to know what a simple model can and cannot do."
Preprocess the dataset: handle missing vitals through imputation, normalize time-series features, and 
encode categorical variables. Then load and fine-tune a pre-trained DNN or shallow feedforward 
network (e.g., via scikit-learn's DNN Classifier or a Keras Sequential model with pre-trained weights 
from a similar medical dataset).

Your implementation must include:

• At least two optimizers (e.g., SGD vs. Adam), plot and compare their loss curves side by side

• Dropout and Batch Normalization as regularization strategies, report the effect of each
Department of Artificial Intelligence Page 2 of 4

• Accuracy, Precision, Recall, and F1-score on the test set

Note: Given the life-critical nature of the task, pay particular attention to Recall in your notebook 
comments. Explain why a false negative is far more dangerous than a false positive in this context.

**Generation 2: Capturing the Patient Timeline**

"A patient's risk is not a snapshot, it is a story told over hours."

Restructure your data as time-series sequences (e.g., hourly vitals windows of 12–24 hours). Load and 
fine-tune a pre-trained LSTM or GRU model (e.g., from PyTorch Hub or a Keras model zoo entry 
trained on sequential medical data, or initialize with pre-trained weights and fine-tune on your dataset).

Your implementation must include:

• A fine-tuned LSTM and a GRU, report performance and training time differences

• A Bidirectional LSTM variant, evaluate whether it improves over the unidirectional version

• Training vs. validation loss curves for each variant

• A notebook comment justifying which architecture is appropriate for real-time monitoring vs. 
retrospective analysis and why

**Generation 3: Reading the Clinical Notes**

"Vitals tell you numbers. Notes tell you the story. The best systems read both."

Load a pre-trained Transformer-based model: BioBERT, ClinicalBERT, or PubMedBERT (all available 
via HuggingFace transformers) and fine-tune it on the free-text clinical notes in your dataset for 
deterioration risk classification.

You do not need to implement attention from scratch. However, your implementation must 
include:

• Correct tokenizer configuration and input formatting for clinical text

• Two fine-tuning strategies: frozen base with trainable head only, and full fine-tuning. Compare 
performance and training cost between both

• Attention weight extraction and visualization from the final attention layer (e.g., using BertViz or 
a manual heatmap), identify which clinical terms the model focuses on most

• A classification head appropriate for binary risk prediction

• A per-class performance breakdown: how well does the model detect true deterioration cases 
vs. stable patients?

Unified Model Comparison

Report all six models in the following table and plot confusion matrices for all models side by side

<img width="1090" height="494" alt="image" src="https://github.com/user-attachments/assets/25d2f0f1-8758-4e0b-8361-f544f1b28ca6" />

