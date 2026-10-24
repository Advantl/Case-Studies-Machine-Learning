# Case-Studies-Machine-Learning
Практические кейсы по машинному обучению на Python (scikit-learn): регрессия, классификация, кластеризация, снижение размерности. В каждом ноутбуке: EDA, подготовка данных, подбор гиперпараметров, оценка и выводы.

## Linear Regression
Датасет: Ames Housing (продажи жилья в Эймсе, Айова, 2006–2010).
- [Dealing with Outliers](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_linear_regression_outliers.ipynb): поиск и удаление 3 аномальных объектов (большая площадь и высокая оценка качества при низкой цене).
- [Dealing with Missing Data](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_linear_regression_missing_data.ipynb): заполнение пропусков с опорой на описание данных (нет подвала/гаража) и средним по району (Lot Frontage), удаление 4 признаков с долей пропусков более 80%.
- [Feature Engineering](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_linear_regression_feature_engineering.ipynb): one-hot кодирование категориальных признаков, включая числовой MSSubClass.
- [Linear Regression Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_linear_regression_project.ipynb): ElasticNet + GridSearchCV, прогноз цены дома. MAE ≈ 14 200 (7,9% от средней цены), RMSE ≈ 20 600 (11,4%).

## Logistic Regression
- [Logistic Regression Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_logistic_regression_project.ipynb): Heart Disease (303 пациента, 13 признаков), наличие болезни сердца. GridSearchCV по C и типу регуляризации; accuracy 0,84, recall по классу «болезнь есть» 0,88 (тест: 31 пациент).

## K-Nearest Neighbors (KNN)
- [K-Nearest Neighbors Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_knn.ipynb): Sonar (208 сигналов по 60 частотам), классификация «мина / камень». Pipeline (StandardScaler + KNN), подбор k; лучший k = 1, accuracy 0,90 (тест: 21 объект).

## Support Vector Machine (SVM)
- [Support Vector Machine Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_svm.ipynb): Wine Fraud (6 497 вин, химические показатели), поиск поддельных. Сильный дисбаланс классов (≈25:1), SVC с class_weight='balanced'; recall по поддельным 0,63 при precision 0,16, вывод о нехватке признаков.

## Decision Tree Classification
- [Decision Tree Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_decision_tree_classification.ipynb): Telco Customer Churn (7 032 клиента), прогноз оттока.
  - Exploratory Data Analysis
  - Анализ оттока клиентов (Customer Churn Analysis): тип контракта, когорты по сроку обслуживания
  - Сравнительный анализ DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier и GradientBoostingClassifier; лучший по F1 для класса «отток» GradientBoosting (F1 0,63, recall 0,75)

## Naive Bayes Classifier
- [Naive Bayes Classifier Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_naive_bayes_classifier.ipynb): Large Movie Review Dataset (2 000 отзывов, после очистки 1 938), тональность текста. Pipeline TF-IDF + MultinomialNB, GridSearchCV; accuracy 0,84.

## K-Means Clustering
- [K-Means Clustering Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_k_means_clustering.ipynb): CIA World Factbook (227 стран, 20 признаков), группировка стран по социально-экономическим показателям. Заполнение пропусков по региону, метод локтя, K = 3, интерактивная карта кластеров.

## DBSCAN
- [DBSCAN Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_dbscan.ipynb): Wholesale Customers (440 клиентов), сегментация по структуре расходов и поиск выбросов. Подбор eps по доле выбросов (eps = 2, min_samples = 16): 2 сегмента и группа крупных нетипичных закупщиков.

## Principal Component Analysis (PCA)
- [Principal Component Analysis Project](https://github.com/Advantl/Case-Studies-Machine-Learning/blob/main/ml_pca.ipynb): рукописные цифры (изображения 8×8, 64 признака). Две главные компоненты объясняют ≈22% дисперсии; лучше всего различимы цифры 2, 4, 6.
