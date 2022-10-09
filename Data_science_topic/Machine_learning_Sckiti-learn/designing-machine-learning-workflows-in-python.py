# Inspect the first few lines of your data using head()
credit.head(3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

    # Inspect the first few lines of your data using head()
credit.head(3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

# Inspect the data types of the columns of the data frame
print(credit.dtypes)


# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)
  # Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2).fit(
  X_train, y_train)

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)

# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2).fit(
  X_train, y_train)

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)

# Assess the accuracy of both classifiers
accuracies['rf'] = accuracy_score(y_test, rf_predictions)

# Set a range for n_estimators from 10 to 40 in steps of 10
param_grid = {'n_estimators': range(10, 50, 10)}

# Optimize for a RandomForestClassifier() using GridSearchCV
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# Define a grid for n_estimators ranging from 1 to 10
param_grid = {'n_estimators': range(1, 11)}

# Optimize for a AdaBoostClassifier() using GridSearchCV
grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# Define a grid for n_neighbors with values 10, 50 and 100
param_grid = {'n_neighbors': [10,50,100]}

# Optimize for KNeighborsClassifier() using GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# Create numeric encoding for credit_history
credit_history_num = LabelEncoder().fit_transform(
  credit['credit_history'])

# Create a new feature matrix including the numeric encoding
X_num = pd.concat([X, pd.Series(credit_history_num)], 1)

# Create new feature matrix with dummies for credit_history
X_hot = pd.concat(
  [X, pd.get_dummies(credit['credit_history'])], 1)

# Compare the number of features of the resulting DataFrames
print(X_hot.shape[1] > X_num.shape[1])

# Function computing absolute difference from column mean
def abs_diff(x):
    return np.abs(x-np.mean(x))

# Apply it to the credit amount and store to new column
credit['diff'] = abs_diff(credit['credit_amount'])

# Create a feature selector with chi2 that picks one feature
sk = SelectKBest(chi2, k=1)

# Use the selector to pick between credit_amount and diff
sk.fit(credit[['credit_amount', 'diff']], credit['class'])

# Inspect the results
sk.get_support()

# Find the best value for max_depth among values 2, 5 and 10
grid_search = GridSearchCV(
  rfc(random_state=1), param_grid={'max_depth': [2, 5, 10]})
best_value = grid_search.fit(
  X_train, y_train).best_params_['max_depth']

# Using the best value from above, fit a random forest
clf = rfc(
  random_state=1, max_depth=best_value).fit(X_train, y_train)

# Apply SelectKBest with chi2 and pick top 100 features
vt = SelectKBest(chi2, k=100).fit(X_train, y_train)

# Create a new dataset only containing the selected features
X_train_reduced = vt.transform(X_train)

### Data fusion

# Group by source computer, and apply the feature extractor 
out = flows.groupby('source_computer').apply(featurize)

# Convert the iterator to a dataframe by calling list on it
X = pd.DataFrame(list(out), index=out.index)

# Check which sources in X.index are bad to create labels
y = [x in bads for x in X.index]

# Report the average accuracy of Adaboost over 3-fold CV
print(np.mean(cross_val_score(AdaBoostClassifier(), X, y)))

# Create a feature counting unique protocols per source
protocols = flows.groupby('source_computer').apply(
  lambda df: len(set(df['protocol'])))

# Convert this feature into a dataframe, naming the column
protocols_DF = pd.DataFrame(
  protocols, index=protocols.index, columns=['protocol'])

# Now concatenate this feature with the previous dataset, X
X_more = pd.concat([X, protocols_DF], axis=1)

# Refit the classifier and report its accuracy
print(np.mean(cross_val_score(
  AdaBoostClassifier(), X_more, y)))
  
  # Create a new dataset X_train_bad by subselecting bad hosts
X_train_bad = X_train[y_train]

# Calculate the average of unique_ports in bad examples
avg_bad_ports = np.mean(X_train_bad['unique_ports'])

# Label as positive sources that use more ports than that
pred_port = X_test['unique_ports'] > avg_bad_ports

# Print the accuracy of the heuristic
print(accuracy_score(y_test, pred_port))

# Fit a Gaussian Naive Bayes classifier to the training data
clf = GaussianNB().fit(X_train, y_train_noisy)

# Report its accuracy on the test data
print(accuracy_score(y_test, clf.predict(X_test)))

# Assign half the weight to the first 100 noisy examples
weights = [0.5]*100 + [1.0]*(len(y_train_noisy)-100)

# Refit using weights and report accuracy. Has it improved?
clf_weights = GaussianNB().fit(X_train, y_train_noisy, sample_weight=weights)
print(accuracy_score(y_test, clf_weights.predict(X_test)))

print(f1_score(y_test, preds))

print(precision_score(y_test, preds))

print((tp + tn)/len(y_test))


# Fit a random forest classifier to the training data
clf = RandomForestClassifier(random_state=2).fit(X_train, y_train)

# Label the test data
preds = clf.predict(X_test)

# Get false positives/negatives from the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

# Now compute the cost using the manager's advice
cost = fp*10 + fn*150

# Score the test data using the given classifier
scores = clf.predict_proba(X_test)

# Get labels from the scores using the default threshold
preds = [s[1] > 0.5 for s in scores]

# Use the predict method to label the test data again
preds_default = clf.predict(X_test)

# Compare the two sets of predictions
all(preds == preds_default)

# Create a range of equally spaced threshold values
t_range = [0.0, 0.25, 0.5, 0.75, 1.0]

# Store the predicted labels for each value of the threshold
preds = [[s[1] > thr for s in scores] for thr in t_range]

# Compute the accuracy for each threshold
accuracies = [accuracy_score(y_test, p) for p in preds]

# Compute the F1 score for each threshold
f1_scores = [f1_score(y_test, p) for p in preds]

# Report the optimal threshold for accuracy, and for F1
print(t_range[argmax(accuracies)], t_range[argmax(f1_scores)])

# Create a scorer assigning more cost to false positives
def my_scorer(y_test, y_est, cost_fp=10.0, cost_fn=1.0):
    tn, fp, fn, tp = confusion_matrix(y_test, y_est).ravel()
    return cost_fp*fp + cost_fn*fn

# Fit a DecisionTreeClassifier to the data and compute the loss
clf = DecisionTreeClassifier(random_state=2).fit(X_train, y_train)
print(my_scorer(y_test, clf.predict(X_test)))

# Refit, downweighting subjects whose weight is above 80
weights = [0.5 if w > 80 else 1.0 for w in X_train.weight]
clf_weighted = DecisionTreeClassifier(random_state=2).fit(
  X_train, y_train, sample_weight=weights)
print(my_scorer(y_test, clf_weighted.predict(X_test)))

################# From workflows to pipelines