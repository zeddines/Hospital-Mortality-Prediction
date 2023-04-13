import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, \
  auc, precision_recall_curve, average_precision_score, \
  ConfusionMatrixDisplay, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import os

# Author: Bertram
# GWTG-HF risk score for COPD
def get_COPD_score(f):
    """
    Parameters:
        f (numpy array): COPD feature vector
    Returns:
        s (numpy array): GWTG-HF risk score vector
    """
    s = []
    for i in f:
        if i == 1:
            s.append(2)
        elif i == 0:
            s.append(0)
        else:
            print("error")
    return np.array(s)

# Author: Bertram
# GWTG-HF risk score for heart rate
def get_hr_score(f):
    """
    Parameters:
        f (numpy array): heart rate feature vector
    Returns:
        s (numpy array): GWTG-HF risk score vector
    """
    s = []
    for i in f:
        if i <= 79:
            s.append(0)
        elif i >= 80 and i <= 84:
            s.append(1)
        elif i >= 85 and i <= 89:
            s.append(3)
        elif i >= 90 and i <= 94:
            s.append(4)
        elif i >= 95 and i <= 99:
            s.append(5)
        elif i >= 100 and i <= 104:
            s.append(6)
        else:
            s.append(8)
    return np.array(s) 

# Author: Bertram
# GWTG-HF risk score for age
def get_age_score(f):
    """
    Parameters:
        f (numpy array): age feature vector
    Returns:
        s (numpy array): GWTG-HF risk score vector
    """
    s = []
    for i in f:
        if i <= 19:
            s.append(0)
        elif i >= 20 and i <= 29:
            s.append(3)
        elif i >= 30 and i <= 39:
            s.append(6)
        elif i >= 40 and i <= 49:
            s.append(8)
        elif i >= 50 and i <= 59:
            s.append(11)
        elif i >= 60 and i <= 69:
            s.append(14)
        elif i >= 70 and i <= 79:
            s.append(17)
        elif i >= 80 and i <= 89:
            s.append(19)
        elif i >= 90 and i <= 99:
            s.append(22)
        elif i >= 100 and i <= 109:
            s.append(25)
        else:
            s.append(28)
    return np.array(s)

# Author: Bertram
# GWTG-HF risk score for blood sodium
def get_sodium_score(f):
    """
    Parameters:
        f (numpy array): blood sodium feature vector
    Returns:
        s (numpy array): GWTG-HF risk score vector
    """
    s = []
    for i in f:
        if i <= 130:
            s.append(4)
        elif i >= 131 and i <= 133:
            s.append(3)
        elif i >= 134 and i <= 136:
            s.append(2)
        elif i >= 137 and i <= 138:
            s.append(1)
        else:
            s.append(0)
    return np.array(s)

# Author: Bertram
# GWTG-HF risk score for blood urea nitrogen
def get_BUN_score(f):
    """
    Parameters:
        f (numpy array): BUN feature vector
    Returns:
        s (numpy array): GWTG-HF risk score vector
    """
    s = []
    for i in f:
        if i <= 9:
            s.append(0)
        elif i >= 10 and i <= 19:
            s.append(2)
        elif i >= 20 and i <= 29:
            s.append(4)
        elif i >= 30 and i <= 39:
            s.append(6)
        elif i >= 40 and i <= 49:
            s.append(8)
        elif i >= 50 and i <= 59:
            s.append(9)
        elif i >= 60 and i <= 69:
            s.append(11)
        elif i >= 70 and i <= 79:
            s.append(13)
        elif i >= 80 and i <= 89:
            s.append(15)
        elif i >= 90 and i <= 99:
            s.append(17)
        elif i >= 100 and i <= 109:
            s.append(19)
        elif i >= 110 and i <= 119:
            s.append(21)
        elif i >= 120 and i <= 129:
            s.append(23)
        elif i >= 130 and i <= 139:
            s.append(25)
        elif i >= 140 and i <= 149:
            s.append(27)
        else:
            s.append(28)
    return np.array(s)

# Author: Bertram
# GWTG-HF risk score for systolic blood pressure
def get_systolic_score(f):
    """
    Parameters:
        f (numpy array): sbp feature vector
    Returns:
        s (numpy array): GWTG-HF risk score vector
    """
    s = []
    for i in f:
        if i <= 59:
            s.append(28)
        elif i >= 60 and i <= 69:
            s.append(26)
        elif i >= 70 and i <= 79:
            s.append(24)
        elif i >= 80 and i <= 89:
            s.append(23)
        elif i >= 90 and i <= 99:
            s.append(21)
        elif i >= 100 and i <= 109:
            s.append(19)
        elif i >= 110 and i <= 119:
            s.append(17)
        elif i >= 120 and i <= 129:
            s.append(15)
        elif i >= 130 and i <= 139:
            s.append(13)
        elif i >= 140 and i <= 149:
            s.append(11)
        elif i >= 150 and i <= 159:
            s.append(9)
        elif i >= 160 and i <= 169:
            s.append(8)
        elif i >= 170 and i <= 179:
            s.append(6)
        elif i >= 180 and i <= 189:
            s.append(4)
        elif i >= 190 and i <= 199:
            s.append(3)
        else:
            s.append(0)
    return np.array(s)

# Author: Bertram
# get GWTG-HF risk score vector based on features X, used as
# benchmark for model comparison 
def get_GWTG_HF_score(X):
    """
    Parameters:
        X (pandas Dataframe): test set features
    Returns:
        sum (numpy array): GWTG-HF risk score vector
    """
    sum = np.zeros(X.shape[0])
    # Systolic BP
    sum += get_systolic_score(X["Systolic blood pressure"].to_numpy())
    # BUN
    sum += get_BUN_score(X["Urea nitrogen"].to_numpy())
    # Sodium
    sum += get_sodium_score(X["Blood sodium"].to_numpy())
    # Age
    sum += get_age_score(X["age"].to_numpy())
    # Heart Rate
    sum += get_hr_score(X["heart rate"].to_numpy())
    # Ethnicity (Missing)
    
    # COPD
    sum += get_COPD_score(X["COPD"].to_numpy())
    return sum

# Author: Bertram
# Model metrics
# Compute f1-score, AUC, AP
# Save Confusion matrix 
# Save ROC curve, PR curve with comparision to GWTG-HF risk score
def test_metrics(name, X, y_t, y_p, y_s = [], path = "results/"):
    """
    Parameters:
        name: name of model
        X (pandas Dataframe): Test set features (not preprocessed)
                              not normalized, used to calculate
                              GWTG-HF risk score
        y_t (pandas Series): Ground truth label for test set
        y_p (numpy array): Model predictions for test set
        y_s (numpy array): Confidence Score for model predictions
        path: relative path for saving figures
    """
    # create folder if not exist
    os.makedirs(path, exist_ok=True)   
    # save confusion Matrix
    cm = confusion_matrix(y_t, y_p, labels = [0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["-", "+"])
    disp.plot()
    plt.savefig(f"{path}{name}, Confusion Matrix.png")
    plt.close()
    # printing f1-score
    print(f"f1-score = {f1_score(y_t, y_p):.2f}")
    
    # get AUC-ROC and AUC-PR comparision if model have
    # scores for each prediction
    if (len(y_s) != 0):
        # get GWTG-HF risk score
        plt.figure()
        gwtg_s = get_GWTG_HF_score(X)
        # ROC curve
        # NOTE ROC curve is not a good metric for inbalance class
        # AUC-ROC curve for our model
        fpr, tpr, thresholds = roc_curve(y_t, y_s, pos_label = 1)
        roc_auc = auc(fpr, tpr)
        print(f"AUC = {roc_auc:.3f}")
        plt.plot(fpr, tpr, color = "red", label = f"Model = {name}, AUC = {roc_auc:.3f}")
        # AUC-ROC curve for GWTG_HF
        fpr, tpr, thresholds = roc_curve(y_t, gwtg_s, pos_label = 1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color = "blue", label = f"Model = GWTG-HF, AUC = {roc_auc:.3f}")
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.savefig(f"{path}{name} vs GWTG-HF, AUC-ROC.png")
        plt.close()
        
        plt.figure()
        # PR curve
        # AUC-PR curve for our model
        precision, recall, _ = precision_recall_curve(y_t, y_s)
        ap = average_precision_score(y_t, y_s)
        print(f"AP = {ap:.3f}\n")
        plt.plot(recall, precision, color = "red", label = f"Model = {name}, AP = {ap:.3f}")
        # AUC-PR curve for GWTF_HF
        precision, recall, _ = precision_recall_curve(y_t, gwtg_s)
        ap = average_precision_score(y_t, gwtg_s)
        plt.plot(recall, precision, color = "blue", label = f"Model = GWTG-HF, AP = {ap:.3f}")
        plt.legend()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR curve")
        plt.savefig(f"{path}{name} vs GWTG-HF, PR.png")
        plt.close()
    