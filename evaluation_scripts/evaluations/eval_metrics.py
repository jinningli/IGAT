from sklearn.metrics import precision_score, recall_score, f1_score
import os
import json
import numpy as np

def precision_purity(label, pred, which):
    l = label[pred == which]
    return np.sum(l == which) / l.shape[0]

def calculate_metrics(ground_truth, predictions):
    
    # Convert "neutral" labels to "opposing"
    # predictions = ["opposing" if label == "neutral" else label for label in predictions]

    # unique_predictions = np.unique(predictions)
    # print("Unique predictions:", unique_predictions)

    # Calculate precision (TODO: Should we use average="macro/weighted/micro"?)
    precision = precision_score(ground_truth, predictions, pos_label="supportive")

    # Calculate recall
    recall = recall_score(ground_truth, predictions, pos_label="supportive")

    # Calculate F1 score
    f1 = f1_score(ground_truth, predictions, average="macro")

    # Calculate accuracy
    accuracy = (ground_truth == predictions).mean()

    oppo_pur = precision_purity(ground_truth, predictions, "opposing")
    sur_pur = precision_purity(ground_truth, predictions, "supportive")
    purity = (oppo_pur + sur_pur) / 2.0

    # KEEP f1, accuracy, purity.
    return precision, recall, f1, accuracy, purity, oppo_pur, sur_pur
