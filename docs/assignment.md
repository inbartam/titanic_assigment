# Data Science Home Assignment

You are given 1 day to complete this assessment.

The task is a **Data Science Task – Classification Model**

The task will be based on the **[Titanic dataset from Kaggle](https://www.kaggle.com/competitions/titanic/data)**.

**Important:** Use **only `train.csv`** from Kaggle. Do **not** use `test.csv` or `gender_submission.csv`. Create your own **train/validation split** from `train.csv` for evaluation.

---

## Classification Task

### Task

Build an end-to-end **classification pipeline** to predict Titanic passenger survival.

### Requirements

* Fetch the **Titanic dataset** directly from Kaggle in your code.
* Load **only `train.csv`** and split it into **train/validation** sets.
* Perform **exploratory data analysis (EDA)** in a **Jupyter Notebook**.
* Apply appropriate **pre-processing** techniques before training.
* Implement and train a **classification model in PyTorch**.

  * The training process must be implemented in a **standalone Python script** (e.g., `train.py`).
  * This script should load the data, preprocess it, train the model, and save the trained weights to disk.
* Evaluate the model on the held-out **validation** split and present the results in **Streamlit**.
* Create an **inference interface** in Streamlit where the user can:

  * Provide the path to a dataset (CSV).
  * Load the trained model from disk (produced by your training script).
  * Run inference and view evaluation results (plots and metrics).

---

## Evaluation Criteria

You will be evaluated on:

* Correct dataset fetching and reproducibility.
* Depth and clarity of EDA.
* Soundness of pre-processing choices.
* Correctness and implementation quality of the PyTorch model.
* Appropriateness of evaluation strategy and clarity of visualizations.
* Functionality and usability of the inference UI.
* Clarity and reproducibility of the training script.

### General

* Code quality, organization, and documentation.
* Clear setup and run instructions.
* Error handling and robustness.
* Originality and problem-solving approach.

---

## Submission Guidelines

* Submit your solution via a **GitHub repository** containing:

  * Source code for both parts.
  * A `data/` folder with a small sample dataset (or link to Titanic dataset on Kaggle).
  * A `README.md` with:

    * Setup instructions
    * Installation commands
    * Run instructions for both apps
    * Example usage (screenshots recommended)
    * Short description of your architecture and design choices

### Installation Example

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

### Run Example

Train the model:

```bash
python train.py
```

Run the app:

```bash
streamlit run ds_app.py
```

---

💡 Good luck! For any questions, feel free to reach out.