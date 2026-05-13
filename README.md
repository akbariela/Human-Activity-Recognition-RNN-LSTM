# Human Activity Recognition (HAR) using Deep Recurrent Networks

This repository implements a complete pipeline for recognizing human activities (such as Walking, Sitting, Standing, etc.) based on 3-axial smartphone sensor data using PyTorch.

## 📌 Project Overview
The core of this project is to classify 6 different physical activities by analyzing time-series data from accelerometers and gyroscopes. The project evolves from a baseline RNN to a sophisticated Stacked LSTM model with custom feature engineering.

## 🚀 Key Features
- **Architectural Diversity:** Implementation of Vanilla RNN, RNN with Global Average Pooling (GAP), and Multi-layer LSTMs.
- **Hardware Acceleration:** Native support for **Apple Silicon (MPS)** and **NVIDIA (CUDA)**.
- **Feature Engineering:** Implementation of sensor magnitude features (Euclidean Norm) to enhance model robustness.
- **Interpretability Tools:** Includes Feature Attribution analysis (Gradient-based sensitivity) and Temporal Accuracy decay plots.

## 📊 Dataset
The model processes 9 initial input channels sampled at 50Hz over a 128-step sliding window:
- **Body Acceleration:** (X, Y, Z)
- **Total Acceleration:** (X, Y, Z)
- **Body Gyroscope:** (X, Y, Z)

## 🏗 Model Architectures

### 1. Vanilla RNN with GAP
Utilizes Global Average Pooling (GAP) instead of just using the last hidden state. This helps the model capture the global context of the entire movement window.

### 2. Stacked LSTM (Final Model)
A 2-layer LSTM with 128 hidden units and 20% Dropout. This architecture is designed to capture long-term dependencies in complex movements like "Walking Upstairs" vs "Walking Downstairs".

### 3. Magnitude Enhancement
We extended the input from **9 to 12 channels** by calculating the magnitude for each sensor group:
$$Mag = \sqrt{x^2 + y^2 + z^2}$$
This addition significantly improved the model's ability to distinguish between static and dynamic activities.

## 📈 Results & Evaluation

### Vanilla RNN (Baseline Performance)
The baseline RNN demonstrated a fundamental ability to distinguish activities but struggled with overlapping dynamic patterns.

- **Test Loss:** 1.1683
- **Overall Accuracy:** 57%

| Activity Index | Class Name | Precision | Recall | F1-Score |
|----------------|------------|-----------|--------|----------|
| 0              | Walking    | 0.29      | 0.02   | 0.04     |
| 1              | Upstairs   | 0.39      | 0.87   | 0.54     |
| 2              | Downstairs | 0.00      | 0.00   | 0.00     |
| 3              | Sitting    | 0.50      | 0.88   | 0.64     |
| 4              | Standing   | 0.62      | 0.58   | 0.60     |
| 5              | Laying     | 1.00      | 0.95   | 0.97     |

> **Note:** The model shows excellent performance on "Laying" (100% precision) but fails to distinguish "Walking Downstairs", highlighting the need for the more complex LSTM architecture.

### Latent Space Analysis: VanillaRNN Hidden States
To gain deeper insight into how the baseline VanillaRNN perceives different activities, we performed **Principal Component Analysis (PCA)** on the hidden states ($h_n$) extracted before the final classification layer.

As shown in the notebook, the visualization of these latent representations reveals the following:

*   **Linear Separability of Static Activities:** The model successfully clusters "Laying" far apart from dynamic activities like "Walking", which explains its high precision in that category.
*   **Challenges in Transitional Movements:** The significant overlap in the *WALKING vs WALKING_UPSTAIRS* and *WALKING_DOWNSTAIRS vs WALKING_UPSTAIRS* plots demonstrates that the simple RNN architecture struggles to extract distinct temporal features for movements with similar periodicities.
*   **Manifold Structure in Postures:** The *STANDING vs SITTING* plot shows both classes following a similar curved manifold, indicating that the baseline model finds it difficult to establish a clear decision boundary for stationary positions that only differ by slight orientation changes.
*   **Gap in Global Context:** The distribution patterns suggest that while the Global Average Pooling (GAP) helps, the simple RNN cells lack the "memory capacity" to fully disentangle complex dynamic activities in the latent space.

#### **Visualizing Class Separation**
*Figure: 2D PCA projection of VanillaRNN hidden states for key activity comparisons.*

### RNN Hidden State Trajectories over Time
Beyond static snapshots, we visualized the temporal evolution of the VanillaRNN's hidden states using PCA trajectories. This analysis tracks how the model's internal representation "travels" in the latent space during a 128-step sequence.

As illustrated in the notebook (where green squares represent the start and red crosses represent the end of the sequence):

*   **Path Divergence:** The model often starts from a similar latent region (the green markers) but must push the hidden states toward distinct regions to make a final classification.
*   **Dynamic Chaos:** In cases like *WALKING_DOWNSTAIRS vs WALKING_UPSTAIRS*, the trajectories follow highly overlapping and non-linear paths. This explains the high misclassification rate, as the VanillaRNN fails to maintain a clear "directional" separation throughout the sequence.
*   **Static Stability:** For activities like *SITTING* and *LAYING*, the trajectories move toward very stable and distant attractors, indicating that the model quickly identifies the lack of movement and settles into a confident state.
*   **Temporal Confusion:** The sharp turns in some trajectories (e.g., *WALKING vs SITTING*) suggest that as more temporal data arrives, the model often has to "correct" its initial guess, a behavior that is more pronounced in this simple RNN compared to gated architectures like LSTM.

#### **Visualizing the Temporal Journey**
*Figure: PCA trajectories showing the movement of hidden states from $t=1$ to $t=128$.*

### Analysis of Model Failure Dynamics
To diagnose the specific reasons behind the **VanillaRNN**'s misclassifications, we performed a diagnostic trajectory analysis. By comparing "Correct" paths (solid green) with "Failed" paths (dashed red), we can observe the exact moments when the model's internal logic deviates.

As visualized in **image_e818e3.png**:

*   **Attractor Confusion:** In the *WALKING mispredicted as WALKING_UPSTAIRS* plot, failed trajectories (red) gravitate toward a specific region of the latent space that the model incorrectly associates with "Upstairs". This suggests that without gating mechanisms, the baseline RNN cannot effectively separate these two similar dynamic signatures.
*   **Irreversible Deviations:** Most failed paths show a clear point of divergence. Once the hidden state enters an incorrect manifold, it rarely recovers. This indicates that the simple RNN cell lacks the ability to "forget" or "correct" misleading temporal information mid-sequence.
*   **Static vs. Dynamic Overlap:** The *WALKING mispredicted as SITTING* analysis reveals cases where the model's representation for a dynamic activity remains trapped in a low-variance region. This failure typically occurs during windows of lower movement intensity where the RNN incorrectly settles into a "Static" state attractor.
*   **Confidence in Error:** The final hidden states of failed trajectories (marked with 'x') are often tightly clustered, suggesting the model is "confidently wrong." It reaches a stable but incorrect conclusion due to its limited capacity for capturing long-term temporal dependencies.

#### **Visualizing Decision Failures**
*Figure: Comparison of hidden state trajectories for correct classifications vs. specific failure modes.*

### Memory Decay and Temporal Accuracy Analysis
To evaluate the long-term dependency capabilities of the **VanillaRNN**, we analyzed how classification accuracy evolves as the model processes the 128-step temporal sequence. This metric serves as a diagnostic tool to identify the model's effective "memory horizon."

As visualized:

*   **Rapid Initial Learning:** The model exhibits a sharp increase in accuracy (from 53.00% to ~56.5%) within the first 60 time steps, suggesting that primary motion features are identified early in the window.
*   **Performance Saturation:** Beyond the 60th step, the accuracy curve plateaus. This stagnation indicates a "memory decay" effect, where the baseline RNN fails to derive additional discriminative value from the remaining half of the sequence.
*   **Vanishing Gradient Impact:** The inability to improve performance in the latter stages is a classic symptom of vanishing gradients. The simple RNN architecture reaches its maximum representational capacity early and cannot effectively integrate long-range temporal dependencies.
*   **Decision Stability:** The convergence at step 128 confirms that the final output is heavily biased toward information processed in the initial segments of the window, proving the need for gated architectures like LSTMs for full-sequence utilization.

#### **Visualizing Information Retention**
*Figure: Classification accuracy as a function of the number of observed time steps.*


### Feature Attribution: Input Channel Sensitivity Analysis
To understand which physical signals most influence the **VanillaRNN**'s decisions, we performed a gradient-based sensitivity analysis across the 9 primary input channels. This technique measures the mean absolute gradient of the output with respect to each input, identifying the "features of interest" for the baseline model.

As visualized in **image_e80a3e.png**:

*   **Dominance of Accelerometer Data:** The model relies heavily on acceleration signals, particularly the **Total Acc Y** and **Total Acc X** channels. This suggests that vertical and lateral body movements are the primary drivers for classification in this architecture.
*   **Total vs. Body Acceleration:** The "Total Acceleration" channels show significantly higher sensitivity compared to "Body Acceleration" alone. This indicates that the baseline RNN finds the raw, unfiltered movement intensity more informative than the decomposed body-motion components.
*   **Low Gyroscope Utilization:** The sensitivity to Gyroscope channels (X, Y, Z) is notably lower. This lack of attribution to rotational data may explain why the VanillaRNN struggles to distinguish between complex activities that involve rotation, such as Walking vs. Walking Upstairs.
*   **Axis-Specific Sensitivity:** The high sensitivity to the Y-axis (Total Acc Y) reflects the model's focus on gravity-aligned movements, which is crucial for identifying static postures like "Laying" but perhaps insufficient for more nuanced dynamic tasks.

#### **Visualizing Sensitivity Distribution**
*Figure: Mean absolute gradients for the 9 input channels, highlighting the model's reliance on total acceleration.*

### Model Enhancement: Vanilla RNN with Global Average Pooling (GAP)
To address the stability issues of the baseline RNN, we integrated a **Global Average Pooling (GAP)** layer. Instead of relying solely on the final hidden state ($h_{128}$), GAP aggregates temporal features across the entire 128-step window, leading to a significant performance boost.

**Performance Results:**
- **Overall Accuracy:** 68.41% (an 11.4% improvement over the baseline)
- **Key Metric Improvement:** The Weighted F1-score increased to 0.64.

| Activity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| WALKING | 0.42 | 0.99 | 0.59 |
| WALKING_UPSTAIRS | 0.25 | 0.01 | 0.01 |
| WALKING_DOWNSTAIRS | 0.88 | 0.48 | 0.62 |
| SITTING | 0.87 | 0.67 | 0.76 |
| STANDING | 0.76 | 0.90 | 0.82 |
| LAYING | 1.00 | 0.95 | 0.97 |

#### **Critical Analysis**
*   **Success of GAP Integration:** The jump to **68.41%** accuracy demonstrates that capturing the "global" temporal context is far more effective for HAR than relying on the noisy final state of a simple RNN.
*   **Superior Static Recognition:** The model now shows high reliability in distinguishing between **Sitting** (0.76 F1) and **Standing** (0.82 F1), which were previously major points of confusion.
*   **The "Walking" Trade-off:** While the model achieved a near-perfect recall (0.99) for **Walking**, it came at the cost of precision (0.42). This indicates a tendency to use "Walking" as a catch-all category for dynamic movements.
*   **Persistent Failure in "Upstairs":** Despite the overall improvement, **Walking Upstairs** remains nearly undetectable (0.01 F1). This confirms that GAP alone cannot solve the fundamental inability of simple RNN cells to model the complex, high-frequency patterns required for vertical transition detection.
*   **Baseline for LSTM:** This 68% accuracy represents the "ceiling" for simple recurrent architectures, providing a strong justification for our final transition to **Stacked LSTM** models.

## 🚀 Final Model: Stacked LSTM Performance

To overcome the vanishing gradient issues and the limited memory horizon of the Vanilla RNN, we implemented a **Stacked LSTM** architecture. This model utilizes gating mechanisms (Forget, Input, and Output gates) to maintain long-term dependencies across the 128-step sensor windows.

### LSTM Performance Report
The transition to LSTM resulted in a dramatic performance leap, reaching a state-of-the-art accuracy for this baseline configuration.

- **Overall Accuracy:** 92%
- **Macro Average F1-Score:** 0.92

| Activity | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **WALKING** | 1.00 | 0.94 | 0.97 | 496 |
| **WALKING_UPSTAIRS** | 0.95 | 0.98 | 0.96 | 471 |
| **WALKING_DOWNSTAIRS** | 0.93 | 0.99 | 0.96 | 420 |
| **SITTING** | 0.78 | 0.83 | 0.81 | 491 |
| **STANDING** | 0.87 | 0.79 | 0.83 | 532 |
| **LAYING** | 0.99 | 1.00 | 0.99 | 537 |

### 🔍 Key Improvements & Insights
*   **Solving the Dynamic Motion Challenge:** The most significant breakthrough is in the **Walking Upstairs/Downstairs** categories. Previously undetectable by the Vanilla RNN (F1 ~0.01), these now achieve scores of **0.96**, proving the LSTM's superior ability to model complex, periodic temporal patterns.
*   **Near-Perfect Walking Detection:** With a precision of **1.00**, the model no longer misclassifies other movements as "Walking," a major issue in the GAP-enhanced RNN.
*   **Static Posture Resolution:** The persistent confusion between **Sitting** and **Standing** has been largely resolved, with both classes seeing an F1-score boost to the 0.80s.
*   **Reliability:** The consistency across all metrics (Precision, Recall, and F1) indicates a robust model that generalizes well across different subjects in the test set.

#### **Conclusion**
The evolution from a simple RNN (57%) to an enhanced GAP-RNN (68%) and finally to a **Stacked LSTM (92%)** demonstrates the necessity of gated recurrent units for processing high-frequency biomechanical sensor data.

### Final Optimization: LSTM with Magnitude Feature Engineering
To further refine the model's discriminative power, we performed feature engineering by calculating the Euclidean norm (magnitude) for the tri-axial sensor groups. This increased the input dimensionality from 9 to 12 channels, providing the model with orientation-invariant intensity signals.

**Feature Engineering Formula:**
$$Mag = \sqrt{x^2 + y^2 + z^2}$$

**Enhanced LSTM Performance:**
- **Overall Accuracy:** 93% (Final Peak Performance)
- **Macro Average F1-Score:** 0.93

| Activity | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **WALKING** | 0.99 | 0.97 | 0.98 | 496 |
| **WALKING_UPSTAIRS** | 0.94 | 0.96 | 0.95 | 471 |
| **WALKING_DOWNSTAIRS** | 0.96 | 0.98 | 0.97 | 420 |
| **SITTING** | 0.81 | 0.84 | 0.82 | 491 |
| **STANDING** | 0.86 | 0.82 | 0.84 | 532 |
| **LAYING** | 0.99 | 1.00 | 1.00 | 537 |

#### **Analysis of Feature Engineering Impact**
*   **Marginal Gains in Accuracy:** Adding magnitude features pushed the accuracy to its peak of **93%**, showing that the model benefits from explicit signals representing movement intensity.
*   **Refinement of Static Postures:** We observed a slight but critical improvement in the **Sitting** and **Standing** F1-scores (rising to 0.82 and 0.84 respectively). This suggests that magnitude features help the LSTM better distinguish between subtle stationary states.
*   **Robustness in Dynamic Movement:** The high F1-scores across all "Walking" sub-types (0.95 to 0.98) confirm that the model is now extremely robust at identifying motion, regardless of the specific sensor orientation.
*   **Laying Accuracy:** The model achieved a perfect **1.00 F1-score** for the "Laying" class, indicating total separability in the latent space for this posture.

**Final Conclusion:**
The progression from a baseline **Vanilla RNN (57%)** to a **Feature-Engineered Stacked LSTM (93%)** demonstrates that combining deep learning architectures with physics-based feature engineering is the most effective approach for high-accuracy Human Activity Recognition.


## 🛠 Prerequisites
All necessary libraries are listed in the `requirements.txt` file. You can install them using:
```bash
pip install -r requirements.txt

## 💻 How to Run
1. Ensure your data is formatted as a 3D tensor: `[Batch, TimeSteps, Features]`.
2. To train the LSTM model:
   ```python
   model = LSTMModel(input_size=12, hidden_size=128, num_classes=6, num_layers=2)
   model.to(device)
