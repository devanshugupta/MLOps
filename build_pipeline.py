import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os
import git
def create_github_repo_and_version_control():
    """Create a GitHub repo, add datasets, and version control."""
    # Step 1: Initialize a local Git repository
    repo_dir = "MLOps Pipeline Abalone"
    os.makedirs(repo_dir, exist_ok=True)
    repo = git.Repo.init(repo_dir)

    # Step 2: Create directories for datasets and reports
    datasets_dir = os.path.join(repo_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # Step 3: Move files to the repo structure
    os.rename("abalone_full_dataset.parquet", os.path.join(datasets_dir, "original_dataset.parquet"))
    os.rename("abalone_splits/train.parquet", os.path.join(datasets_dir, "train.parquet"))
    os.rename("abalone_splits/test.parquet", os.path.join(datasets_dir, "test.parquet"))
    os.rename("abalone_splits/production.parquet", os.path.join(datasets_dir, "production.parquet"))
    os.rename("abalone_dataset_profile.html", os.path.join(repo_dir, "abalone_dataset_profile.html"))

    # Step 4: Add files to the Git repository
    repo.index.add([os.path.join(datasets_dir, "train.parquet"),
                    os.path.join(datasets_dir, "test.parquet"),
                    os.path.join(datasets_dir, "production.parquet"),
                    os.path.join(datasets_dir, "original_dataset.parquet"),
                    os.path.join(repo_dir, "abalone_dataset_profile.html")])

    # Step 5: Commit the changes
    repo.index.commit("Initial commit: Added datasets and profiling report")

    print(f"GitHub repo initialized and files committed in: {repo_dir}")

def build_ml_pipeline():
    """Create an ML pipeline with Scikit-Learn."""
    # Step 1: Load dataset from GitHub raw link
    github_raw_url = "https://raw.githubusercontent.com/your_username/your_repo/main/datasets/train.parquet"  # Replace with your raw file link
    df = pd.read_parquet(github_raw_url)

    # Step 2: Separate features and target
    X = df.drop("Rings", axis=1)
    y = df["Rings"]

    # Step 3: Define transformations for numerical and categorical features
    numerical_features = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]
    categorical_features = ["Sex"]

    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Step 4: Create a pipeline with a model
    model = RandomForestRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Step 5: Evaluate the pipeline
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_squared_error")
    print("Model Mean Squared Error:", -scores.mean())

if __name__ == "__main__":
    # Create GitHub repo and version control files
    create_github_repo_and_version_control()

    # Build and evaluate the ML pipeline
    build_ml_pipeline()
