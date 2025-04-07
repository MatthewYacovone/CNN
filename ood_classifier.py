import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('testing123.csv', engine='python')
print(df.head(5))
df.columns = ['image_idx', 'polarization', 'in_distribution', 'correct']
df.drop(columns=['image_idx'])

def train_NB():
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
    train_NB()


