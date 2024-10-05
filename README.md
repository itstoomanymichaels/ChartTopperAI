# Team 125: ChartTopperAI


Project Proposal
---

[Team 125 Timeline and Responsibility Chart](https://docs.google.com/spreadsheets/d/1BAeCRSATNG66czkyHXANuw6V8RqQEYm4ZzUVjBAZx54/edit?usp=sharing)

## Proposal Contributions

|Member               |Contribution            |
|---------------------|------------------------|
|Eric Guenoun         |Gantt Chart             |
|Vinay Halwan         |Proposal Section 2 & 4  |
|Tripp Hanley         |Proposal Section 1      |
|Brandon Harris       |Presentation & Video    |
|Michael Herndon III  |Proposal Section 3      |


## Introduction & Background

For our project, we will create an accessible model for a song’s outreach and popularity through audio features and marketing. Some experimental models exist already, such as an MLP finding a statistically significant correlation with loudness and duration [[1]](#references). Other studies found that audio features are only a small factor, as an audio-based GNB model only had 60% accuracy, identifying artist and genre data as a potential future feature [[2]](#references). For our training data, Spotify’s existing API analyzes and provides audio data such as ‘danceability’ and tempo scores along with genre data that can be compiled to a dataset for our model to learn and use. We will also follow Reiman and include genre and artist data to explore if genre changes success rate.

## Problem Definition

Predicting a music’s success is incredibly difficult within the increasingly complex, fast-paced music industry landscape that exists today. Many current models are dependent on subjective evaluations, thereby creating erratic results and sidelining emerging talent. The unpredictability can pose challenges to artists striving to reach the audience effectively. As streaming platforms are transforming the landscape of music, a data driven approach is important for understanding the interplay of various factors like audio characteristics and listener preferences. Using machine learning techniques will allow us to increase prediction accuracy and foster a diverse space for artists seeking recognition.

## Methods

To build an accurate model for predicting song success on streaming platforms, effective data preprocessing is essential. First, we plan to standardize features like play count, likes, and shares using Scikit-learn’s StandardScaler, ensuring that the varying scales of these features do not negatively impact the model’s performance. For categorical variables such as song genre, artist name, and country of origin, we will employ LabelEncoder to convert them into numerical representations that can be used by machine learning algorithms. Additionally, to avoid overfitting and improve performance, we will apply SelectKBest to reduce the dimensionality of the dataset by selecting the most relevant features.

After preprocessing, we intend to experiment with several machine learning models. Linear regression will serve as a baseline for predicting continuous targets like the number of streams. Random forests, implemented through RandomForestClassifier, are well-suited for our data because they can handle both numerical and categorical variables without explicit encoding [[3]](#references). To further improve accuracy, especially in datasets where relationships are non-linear, we will use XGBoost, which builds trees sequentially to correct errors made by earlier trees [[4]](#references).

For our supervised learning approach, we plan to focus primarily on ensemble methods, which are especially useful for handling complex, non-linear data [[5]](#references) [[6]](#references). In addition, we will use logistic regression for binary classification tasks, such as predicting whether a song will become a hit or not. This combination of preprocessing and model experimentation should provide a strong foundation for predicting song success.

## Results and Discussion

The primary goal of this project is to create a machine learning model that’s capable of predicting the success of a song. It’ll utilize key metrics like accuracy, f1 score, and area under the ROC curve. We desire in achieving an f1 score of at least 0.8, and an AUC-ROC score that exceeds 0.85 to show a strong balance between recall and precision as to highlight the model’s abilities to distinguish between successful and unsuccessful tracks. We are also committed towards ethical considerations by addressing biases of genre, artist popularity, and other subjective factors. Our expectation is that the model will deliver a high predictive accuracy while encouraging a range of artistic expression.

## Video and Presentation
- [Video](https://youtu.be/5eBhSzQKD0U)
- [Presentation](https://docs.google.com/presentation/d/18zIXuh5MFcKSHXZa84NbpHkGcrnm5MaotNsH93NmArI/edit?usp=sharing)

## References

[1] “[PDF] Collaboration-Aware Hit Song Prediction | Semantic Scholar,” Semanticscholar.org, 2023. https://www.semanticscholar.org/reader/fe52aade53722e81d8dda2e0093e0ef4b9d56799 (accessed Oct. 04, 2024).

[2] M. Reiman, P. Örnell, K. Skolan, F. Elektroteknik, and O. Datavetenskap, “Predicting Hit Songs with Machine Learning.” Accessed: Aug. 02, 2022. [Online]. Available: https://kth.diva-portal.org/smash/get/diva2:1214146/FULLTEXT01.pdf

[3] L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001, doi: https://doi.org/10.1023/a:1010933404324.

[4] T. Chen and C. Guestrin, “XGBoost: a Scalable Tree Boosting System,” Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’16, pp. 785–794, 2016., doi: https://dl.acm.org/doi/pdf/10.1145/2939672.2939785

[5] T. G. Dietterich, “Ensemble Methods in Machine Learning,” Multiple Classifier Systems, vol. 1857, pp. 1–15, 2000, doi: https://doi.org/10.1007/3-540-45014-9_1.

[6] Z. Zhou, “Ensemble Methods: Foundations and Algorithms,” 2012. https://tjzhifei.github.io/links/EMFA.pdf (accessed Oct. 04, 2024).
‌
‌
‌
‌
