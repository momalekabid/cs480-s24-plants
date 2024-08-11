# CS 480 Aug 10 2024, Mohamed Malek Abid Final Project
# Student ID: 20873952
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib
from sklearn.metrics import r2_score
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


# extract features using the DINOv2 model
def extract_dinov2_features(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).squeeze().cpu().numpy()
    return features


# load the pretrained DINOv2 model
def get_dinov2_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg_lc")
    model.eval()
    return model


# setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_model = get_dinov2_model().to(device)

transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),  # DINOv2 typically uses 224x224 images, so we upsample
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# load into df
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# identify feature columns and targets
feature_columns = train_df.columns[:164]
# swapping columns X26 and X50
target_columns = [
    "X4_mean",
    "X11_mean",
    "X18_mean",
    "X50_mean",
    "X26_mean",
    "X3112_mean",
]

# paths to save/load features
train_image_features_path = "train_image_features_dinov2.npy"
test_image_features_path = "test_image_features_dinov2.npy"

if os.path.exists(train_image_features_path) and os.path.exists(
    test_image_features_path
):
    # Load precomputed features
    train_image_features = np.load(train_image_features_path)
    test_image_features = np.load(test_image_features_path)
else:
    train_image_features = []
    test_image_features = []

    train_image_dir = "data/train_images/"  # Directory containing train images
    test_image_dir = "data/test_images/"  # Directory containing test images

    # extract features for train images
    for img_id in tqdm(train_df["id"], desc="Extracting Train Image Features"):
        img_path = os.path.join(train_image_dir, f"{img_id}.jpeg")
        features = extract_dinov2_features(img_path, dinov2_model, transform, device)
        train_image_features.append(features)

    # extract features for test images
    for img_id in tqdm(test_df["id"], desc="Extracting Test Image Features"):
        img_path = os.path.join(test_image_dir, f"{img_id}.jpeg")
        features = extract_dinov2_features(img_path, dinov2_model, transform, device)
        test_image_features.append(features)

    train_image_features = np.array(train_image_features)
    test_image_features = np.array(test_image_features)

    np.save(train_image_features_path, train_image_features)
    np.save(test_image_features_path, test_image_features)

# convert features to DataFrame
train_image_features_df = pd.DataFrame(train_image_features, index=train_df.index)
test_image_features_df = pd.DataFrame(test_image_features, index=test_df.index)

# concatenate features with existing data
train_df = pd.concat([train_df, train_image_features_df], axis=1)
test_df = pd.concat([test_df, test_image_features_df], axis=1)

# update feature columns to include DINOv2 features
feature_columns = list(feature_columns) + list(train_image_features_df.columns)

# check to ensure all feature column names are strings
train_df.columns = train_df.columns.astype(str)
test_df.columns = test_df.columns.astype(str)
feature_columns = [str(col) for col in feature_columns]

# splitting the data
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_columns], train_df[target_columns], test_size=0.1, random_state=777
)

# standard regression feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df[feature_columns])

# initialize CatBoostRegressor with GPU support
catboost_regressor = CatBoostRegressor(
    loss_function="MultiRMSE",
    iterations=3000,
    depth=9,
    learning_rate=0.05,
    task_type="GPU",
    devices="0",
    boosting_type="Plain",
    verbose=200,
)

# fit the model
catboost_regressor.fit(
    X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), early_stopping_rounds=100
)

# saving the model
joblib.dump(catboost_regressor, "catboost_multioutput_model_catboost.pkl")

# validation predictions and scoring
val_predictions = catboost_regressor.predict(X_val_scaled)
r2_scores = r2_score(y_val, val_predictions, multioutput="raw_values")
print(f"CatBoost R2 scores for each target: {dict(zip(target_columns, r2_scores))}")
average_r2_score = np.mean(r2_scores)
print(f"CatBoost Average R2 score: {average_r2_score:.4f}")

# test predictions and submission file creation
predictions = catboost_regressor.predict(X_test_scaled)
submission_df = pd.DataFrame(
    predictions, columns=[col.replace("_mean", "") for col in target_columns]
)
submission_df.insert(0, "id", test_df["id"].astype(int))
submission_df = submission_df.head(6391)

# swap the columns X26 and X50 contents
submission_df["X26"], submission_df["X50"] = (
    submission_df["X50"],
    submission_df["X26"].copy(),
)
submission_df.to_csv("submission.csv", index=False)
print("Submission file is ready.")
