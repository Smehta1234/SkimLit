# SkimLit

SkimLit is a project that leverages machine learning techniques to analyze and visualize research paper abstracts. The project is built around the **PubMed RCT** dataset, with a focus on automating the classification of sentences in medical abstracts using various models.

## Dataset

The project utilizes the **PubMed RCT** dataset, which contains structured abstracts from PubMed articles categorized into different sections, such as Background, Objectives, Methods, Results, and Conclusions.

## Project Overview

The main objectives of the SkimLit project are to:
- Preprocess and analyze the **PubMed RCT** dataset.
- Perform exploratory data analysis and visualization.
- Implement various feature engineering techniques, including **one-hot encoding** and **embedding layers**.
- Train and evaluate machine learning models for classifying abstract sentences.

## Models Implemented

### 1. Baseline Model - Naive Bayes (Model 0)

- The baseline model is implemented using **Multinomial Naive Bayes**, which relies on traditional machine learning techniques.
- Feature extraction involves converting text data into numerical form using techniques such as **TF-IDF**.
- Provides a benchmark for evaluating deep learning models.

### 2. Deep Learning Model - Conv1D with Token Embeddings (Model 1)

- A more advanced deep learning approach using **1D Convolutional Neural Networks (Conv1D)**.
- Token embeddings are used to represent words, allowing the model to capture contextual information.
- The model architecture includes embedding layers, convolutional layers, and fully connected layers for classification.
- Achieves improved performance compared to the baseline model.


## Dependencies

Ensure you have the following dependencies installed before running the project:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Results

- The baseline Naive Bayes model serves as a quick and interpretable benchmark with a accuracy of 72%
- The Conv1D model with token embeddings demonstrates improved performance, leveraging deep learning capabilities with a accuracy of 79%

## Future Work

- Explore transformer-based models like BERT for improved text understanding.
- Experiment with additional feature engineering techniques.
- Fine-tune hyperparameters for better accuracy and generalization.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License.

---

**Author:** Sanchit Mehta 
**GitHub:** (https://github.com/Smehta1234)

