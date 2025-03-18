### Exploratory Lab Report: Handwritten Digit Classification Using Machine Learning

#### Introduction
The MNIST dataset consists of 28x28 pixel grayscale images representing handwritten digits from 0 to 9. Each image is transformed into a 784-dimensional vector, where pixel values range between 0 and 255. The dataset is provided in separate training (mnist_train.csv) and testing (mnist_test.csv) files. In this study, both datasets were combined into a single DataFrame to facilitate preprocessing, ensuring missing values were properly handled before re-splitting the data into training and test sets.

#### Methodology
##### Dataset Preparation
- **Data Integration**: The training and testing datasets were merged into one to allow uniform preprocessing.
- **Normalization & Standardization**: Pixel intensity values were scaled to the [0,1] range and further standardized using (`StandardScaler`).
- **Feature Selection**: `PSA` The top 95% most relevant features were chosen using the ANOVA F-test via Principal Statistical Analysis (PSA).
- **Data Splitting**: The refined dataset was divided into training (80%) and testing (20%) subsets, maintaining class balance through stratification.

##### Models Used
Four models were trained:
1. **Logistic Regression**: `max_iter=50`.
2. **K-Nearest Neighbors (KNN)**: Default parameters with k=5.
3. **Naive Bayes**: Gaussian Naive Bayes.
4. **Neural Network (MLP)**: A basic multi-layer perceptron (MLP) with one hidden layer containing 100 neurons.

#### ** Neural Network (MLP)**
- **Hyperparameters Tuned:**
ANN performance was fine-tuned using:
  - `hidden_layer_sizes`: [(256, 128), (128, 128, 64), (256, 128, 64)]
  - `max_iter`: 50
  - **Accuracy:** 0.912

- **Best Results:**
  - `hidden_layer_sizes`: [512, 256, 128]
  - `max_iter`: 50
- **Accuracy:** 0.920

##### Model Evaluation Metrics
The performance of each model was evaluated using the following metrics:
- **Accuracy:** Percentage of correctly classified samples.
- **Precision:** Ratio of true positives to the total predicted positives.
- **Recall:** Ratio of true positives to the total actual positives.
- **F1-Score:** Harmonic mean of precision and recall.

Confusion matrices were also plotted to visualize the misclassifications.

#### Results
| Model                      | Accuracy  |
|----------------------------|-----------|
| Logistic Regression        | 0.919     |
| KNN                        | 0.948     |
| Naive Bayes                | 0.459     |
| ANN                        | 0.971     |

- **Logistic Regression**: Moderate accuracy (0.919), impacted by its assumption that features are independent, which is not entirely valid for image data.
- **KNN**: 0.948, strong distance-based classification, benefiting from feature selection.
- **Naive Bayes**: Poor accuracy (0.459), likely due to its inability to capture the dataset's complexity.
- **ANN**: Strong accuracy (0.971) with a simple architecture

**Visualizations**:
- A bar chart depicted the comparative performance of all models, highlighting the tuned ANN as the best performer.
- Confusion matrices provided insights into common classification errors (e.g., misclassification between 5 and 3).

#### Discussion
The optimized ANN outperformed other models with a peak accuracy of 0.971, effectively capturing non-linear relationships in the dataset. Combining the training and testing sets enabled consistent NaN handling, although dropping samples with missing labels slightly reduced the dataset size. The feature selection process, which retained 200 dimensions, preserved essential information while reducing complexity, benefiting distance-based models like KNN. Logistic Regression struggled significantly, likely due to its linear nature and the reduced feature set. Naive Bayes exhibited reasonable performance but was limited by its assumption of independent features. Preprocessing played a crucial role in model success, and future studies could investigate deeper neural network architectures or alternative feature selection methods to enhance classification accuracy.

#### Conclusion
This experiment explored handwritten digit classification through various machine learning models. By consolidating the dataset and implementing robust preprocessing techniques, the ANN model achieved an accuracy of 0.971, outperforming other models. The study emphasized the impact of data preparation, feature selection, and model choice on classification performance, offering a structured approach for future applications in pattern recognition.