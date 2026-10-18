import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train():
    # load data
    df = pd.read_csv("student_exam_score_dataset.csv")

    # split features and target
    x = df.drop("exam_score", axis=1)
    y = df["exam_score"]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    #  train model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # save model
    joblib.dump(model, "student_score_model.joblib")


if __name__ == "__main__":
    train()
