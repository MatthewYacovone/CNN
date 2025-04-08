import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_classifier(disagreement_df):
    disagreement_df.columns = ['image_idx', 'polarization', 'in_distribution', 'correct']
    df = disagreement_df.drop(columns=['image_idx'])
    X = df[['polarization', 'correct']]
    y = df['in_distribution']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)


    y_predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predictions)
    cm = confusion_matrix(y_test, y_predictions)
    report = classification_report(y_test, y_predictions)

    print(f'Naive Bayes Classifier Accuracy; {(accuracy * 100):2f}')
    print('Confusion Matrix')
    print(cm)
    print('Classification Report')
    print(report)

    return clf

if __name__ == '__main__':
    df = pd.read_csv('emsemble_disagreement_from_cnn_ensemble.csv')
    run_classifier(df)

