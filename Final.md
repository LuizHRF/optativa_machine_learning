# Random Forest models

## 1. What are Random Forests?

Random Forests are **ensemble learning methods** that combine multiple **Decision Trees** to improve prediction performance and reduce overfitting.

- Each tree is trained on a different **subset of the data**.
- The final prediction is based on the **majority vote** (for classification) or **average prediction** (for regression).
- They’re robust, work well out-of-the-box, and handle both classification and regression problems.

## 2. How it works

Random Forest comes in two main types:

- **RandomForestClassifier**: used when the target variable is categorical (e.g., spam detection).
- **RandomForestRegressor**: used when the target variable is numerical (e.g., predicting prices).

Despite the difference in output, both models work similarly under the hood:
- Each tree is trained on a random sample (with replacement) of the training data.
- At each split, a random subset of features is selected to find the best split.
- Predictions from all trees are combined for a final output.

The key word for Random Forest models is **Diversity**. We want the most diverse set of trees in our forest so the final result will be more accurate and will have considered many different features and examples under different scopes.


## 3. Subsampling (Bootstrap Aggregation)

Each tree in the forest is trained on a **bootstrap sample**:
- A random sample **with replacement** from the original dataset.
- Same number of rows as the original data, but some rows may appear multiple times, others not at all.
- This technique introduces variability between trees, which helps reduce overfitting.

The rows *not* included in a tree’s training sample are called **out-of-bag (OOB)** data, which can be used for validation.


## 4. Feature Bagging (Random Subset of Features)

To increase diversity even further, Random Forest uses **feature bagging**:
- At each split in a tree, only a **random subset of features** is considered.
- This ensures that not all trees rely on the same features or patterns.
- In classification, the default is √(number of features); in regression, it's typically (number of features)/3.

This helps the model learn different perspectives from the data, improving generalization.


## 5. Important Hyperparameters

Some key parameters that control the behavior of a Random Forest:

| Parameter         | Description |
|------------------|-------------|
| `n_estimators`   | Number of trees in the forest (more = better, up to a point) |
| `max_depth`      | Maximum depth of each tree (limits overfitting) |
| `max_features`   | Number of features to consider at each split |
| `oob_score`      | Whether to calculate out-of-bag validation score |


## 6. Random Forest-Specific Tools and Insights

While generic metrics like accuracy or mean squared error apply to any model, Random Forest offers **unique built-in tools and insights** that help us better understand and improve the model:

### Feature Importance attribute

- **What it is**: A score that shows how much each feature contributes to the model's decisions.
- **How it’s calculated**: Based on how much each feature reduces impurity (like Gini or entropy) across all the trees and splits.
- **Why it matters**: 
  - Helps with **interpretability** — you can see which features are driving your model's predictions.
  - Useful for **feature selection** — you can eliminate low-importance features to simplify your model without losing performance.
- **Note**: Feature importance doesn't show direction (positive/negative), only relevance.

> In sklearn, it's available as `.feature_importances_` and can be plotted.

---

### Out-of-Bag (OOB) Score

- **What it is**: A built-in validation technique that uses the data *left out* during bootstrap sampling to evaluate model performance.
- **How it works**: Each tree sees only ~63% of the training data. The remaining ~37% (the “out-of-bag” data) is used to test that tree’s predictions.
- **Why it's useful**:
  - Gives you a **reliable estimate of generalization error** without needing a separate validation set or cross-validation.
  - Saves time and simplifies workflow.

> Enable it by setting `oob_score=True` when creating your RandomForest model.

---

### Bias-Variance Behavior and Overfitting

- Random Forests reduce **variance** by averaging many trees, which helps prevent overfitting.
- But they’re not immune — they can still **overfit** if:
  - Trees are too deep.
  - Not enough randomness (e.g., high `max_features`).
  - Training data is small or noisy.

**How to control it:**
- Use parameters like:
  - `max_depth` to limit how deep trees can grow.
  - `min_samples_split` or `min_samples_leaf` to avoid over-specialization.
  - `max_features` to inject more randomness.
