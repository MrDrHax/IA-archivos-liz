import numpy as np
from sklearn.naive_bayes import GaussianNB

def trainAndTest(x_train, y_train, x_test, y_test):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    model = GaussianNB()

    model.fit(x_train, y_train)

    # Step 4: Make predictions
    predictions = model.predict(x_test)

    correct = 0
    # Output the predictions
    for i, prediction in enumerate(predictions):
        if prediction == y_test[i]:
            correct += 1

    print(f"accuracy: {correct}/{len(predictions)} ({correct/len(predictions)}%)")
