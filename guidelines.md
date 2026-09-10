# Project title: Boosting and Additive Models


## Evaluation: criteria and timeline
The evaluation of the project will mostly focus on the **correctness** and **completeness** of the methodology employed, rather than on the accuracy of the trained model(s) or training time. The **reproducibility** of the results and the **quality of the report** also play a substantial role in the final evaluation.
Within approximately 3 to 4 weeks after each deadline, the submissions will be reviewed and the students will be contacted to set the date of an oral examination, usually held online but possibly in person upon request, where the project will be thoroughly discussed with the TAs; students are required to prepare a brief presentation (~5/10 mins) of their work (without slides), they will be asked to motivate the choices made in the assignment and discuss the results.
We stress that group projects are not allowed: students must complete their projects individually.

## Coding and report specifications
The preferred language is **Python**, any other choice must be agreed upon with the TAs. Jupyter notebooks are allowed. All the required algorithms, unless specified otherwise, must be **implemented from scratch**, external libraries like **numpy** and **matplotlib** may be used **to only ease computation and plotting results.** The submitted code should be reproducible and run in reasonable time (see the datasets section).

The report must be written in LaTeX and the submitted repository must contain the final PDF. The report should describe the work done to complete the assignment, discussing in particular the choices made regarding the datasets, the implementation, and the visualizations. Furthermore, the report should provide a thorough analysis of the results, highlighting both the positive and the negative aspects. The report should be 5-10 pages long.

## Datasets
The assignments do not specify the exact dataset for the task, instead you are required to pick or generate datasets and motivate the associated choices in the report. **All the listed types of dataset must be included in the project, the report must include a comparison of the results across the datasets.**

For **real-world datasets,** standard sources such as the **UCI Machine Learning Repository** or **sklearn.datasets** are recommended. When selecting a dataset, make sure it is consistent with the task at hand, using classification datasets for classification problems and regression datasets for regression problems.

For **synthetic datasets,** you are **free to design the data-generating process.** A typical approach is to sample inputs from a chosen distribution (e.g., Gaussian or uniform distribution), possibly in multiple dimensions, and then define the labels as a deterministic function of the inputs. For instance, a linear function can be used to generate linearly separable data. Noise can be added to make the problem more realistic and to study robustness.

**When applicable, datasets should be split into training and test sets** and special care must be taken to ensure that **no data leakage occurs.** In these cases, the model must be trained exclusively on the training set and performance should be evaluated on the test set. If a different evaluation protocol is more appropriate for a specific assignment, it should be clearly justified.

The running time of the algorithms is not a primary concern for this project, avoid excessively large datasets and prefer small- to medium-scale problems that allow you to run multiple experiments efficiently.


## Assignment Specifications

*A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*

Yoav Freund, Robert Schapire
Journal of Computer and System Sciences, 1997

In the course, you have studied empirical risk minimization and basic predictors such as linear models and trees. These methods rely on choosing a single model from a hypothesis class, possibly with regularization. However, a different approach is to combine many simple models into a stronger one.

Boosting is a method that builds a strong classifier by iteratively combining weak learners, typically simple models that perform only slightly better than random guessing. At each step, the algorithm focuses on the examples that are hardest to classify, gradually improving performance. This process can be interpreted both as an optimization procedure and as a way of constructing additive models.

This assignment explores boosting from both perspectives. You will implement and analyze AdaBoost, and study how combining weak learners affects training error, test error, and robustness. This connects to the course material on loss minimization and extends it to iterative, adaptive model construction.


### Objective
- Understand how boosting combines weak learners
- Study the relationship between training error and generalization

### Required dataset
- Real-world classification dataset
- Synthetic classification dataset
### Tasks
- Implement AdaBoost with:
  - Decision stumps as weak learners
- Track over iterations:
  - Training error
  - Test error
- Analyze:
  - Evolution of sample weights
  - Contribution of weak learners
- Compare (you can use off-the-shelf solutions):
  - Single decision tree
  - Logistic regression

### Expected output
- Training error decreasing over iterations
- Non-trivial behavior of test error
- Insight into how boosting focuses on hard examples

### Extensions
- Study overfitting as number of rounds increases
- Compare with gradient boosting (optional)
- Analyze margin distribution






