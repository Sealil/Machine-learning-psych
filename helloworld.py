
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree


file_path = r'C:\Users\lg\OneDrive\Desktop\data-final.csv'
try:
    df = pd.read_csv(file_path, sep='\t')
    print("CSV file read successfully.")
except pd.errors.ParserError as e:
    print(f"Error while reading CSV file: {e}")


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999, inplace=True)

x_columns = ['EXT2', 'EXT3', 'EXT4']

# Column name to use as Y matrix
y_column = 'EXT5'

# Extract X matrix and Y matrix
X = df[x_columns].values
Y = df[y_column].values

print(X[0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
mlp = MLPClassifier(max_iter=300).fit(X_train, Y_train)
mlp1 = MLPClassifier(max_iter=500).fit(X_train, Y_train)

y_pred1 = mlp1.predict(X_test)

count1 = 0

for i in range (len(y_pred1)):
    if y_pred1[i] == Y_test[i]:
        count1 += 1

print("MLP classifier 1 accuracy:")
print(count1/len(Y_test))

y_pred_train1 = mlp1.predict(X_train)
count_train1 = 0

for i in range (len(y_pred_train1)):
    if y_pred_train1[i] == Y_train[i]:
        count_train1 += 1
print("MLP classifier 1 TRAINING accuracy:")
print(count_train1/len(Y_train))


y_pred = mlp.predict(X_test)

count = 0

for i in range (len(y_pred)):
    if y_pred[i] == Y_test[i]:
        count += 1

print("MLP classifier 2 accuracy:")
print(count/len(Y_test))


y_pred_train = mlp.predict(X_train)
count_train = 0

for i in range (len(y_pred_train)):
    if y_pred_train[i] == Y_train[i]:
        count_train += 1
print("MLP classifier 2 TRAINING accuracy:")
print(count_train/len(Y_train))

#print(X_train[0])
#print(Y_train[0])



decision_trees = RandomForestClassifier(max_depth=5)
decision_trees.fit(X_train, Y_train)

count_tree = 0
count_tree_train = 0
y_pred_tree_train = decision_trees.predict(X_train)

for i in range (len(y_pred_tree_train)):
    if y_pred_tree_train[i] == Y_train[i]:
        count_tree_train += 1

print("Decision Tree TRAINING accuracy:")
print(count_tree_train/len(Y_train))

y_pred_tree = decision_trees.predict(X_test)

for i in range (len(y_pred_tree)):
    if y_pred_tree[i] == Y_test[i]:
        count_tree += 1

print("Decision Tree accuracy:")
print(count_tree/len(Y_test))



clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, Y_train)

count_clf_train = 0
y_pred_clf_train = clf.predict(X_train)

for i in range (len(y_pred_clf_train)):
    if y_pred_clf_train[i] == Y_train[i]:
        count_clf_train += 1

print("CLF TRAINING accuracy:")
print(count_clf_train/len(Y_train))

y_pred_clf = clf.predict(X_test)

count_clf = 0

for i in range (len(y_pred_clf)):
    if y_pred_clf[i] == Y_test[i]:
        count_clf += 1

print("CLF accuracy:")
print(count_clf/len(Y_test))



mlp_results = []
tree_results = []
mlp1_results = []
clf_results = []


for i in range (len(y_pred)):
    if y_pred[i] == Y_test[i]:
        mlp_results.append("true")
    else: 
        mlp_results.append("false")

for i in range (len(y_pred_tree)):
    if y_pred_tree[i] == Y_test[i]:
        tree_results.append("true")
    else: 
        tree_results.append("false")

for i in range (len(y_pred1)):
    if y_pred1[i] == Y_test[i]:
        mlp1_results.append("true")
    else: 
        mlp1_results.append("false")

for i in range (len(y_pred_clf)):
    if y_pred_clf[i] == Y_test[i]:
        clf_results.append("true")
    else: 
        clf_results.append("false")

count_both_correct = 0

for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "true":
        if mlp1_results[i] == "true" and clf_results[i] == "true":
            count_both_correct += 1

print("Percentage of answers both model predicted correct:")
print(count_both_correct/len(mlp_results))

count_only_mlp = 0
count_only_mlp1 = 0
count_only_tree = 0
count_only_clf = 0

for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "false":
        if mlp1_results[i] == "false" and clf_results[i] == "false":
            count_only_mlp += 1

for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "false" and tree_results[i] == "false":
        if mlp1_results[i] == "true" and clf_results[i] == "false":
            count_only_mlp1 += 1

for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "false" and tree_results[i] == "true":
        if mlp1_results[i] == "false" and clf_results[i] == "false":
            count_only_tree += 1

for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "false" and tree_results[i] == "false":
        if mlp1_results[i] == "false" and clf_results[i] == "true":
            count_only_clf += 1

print("Percentage only MLP predicted correct:")
print(count_only_mlp/len(mlp_results))
print("Percentage only MLP1 predicted correct:")
print(count_only_mlp1/len(mlp_results))
print("Percentage only Tree predicted correct:")
print(count_only_tree/len(mlp_results))
print("Percentage only CLF predicted correct:")
print(count_only_clf/len(mlp_results))

count_shared = 0
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "true":
        if mlp1_results[i] == "false" and clf_results[i] == "false":
            count_shared += 1
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "false":
        if mlp1_results[i] == "true" and clf_results[i] == "false":
            count_shared += 1
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "false":
        if mlp1_results[i] == "false" and clf_results[i] == "true":
            count_shared += 1
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "true":
        if mlp1_results[i] == "false" and clf_results[i] == "true":
            count_shared += 1
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "true":
        if mlp1_results[i] == "true" and clf_results[i] == "false":
            count_shared += 1
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "true" and tree_results[i] == "false":
        if mlp1_results[i] == "true" and clf_results[i] == "true":
            count_shared += 1

for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "false" and tree_results[i] == "true":
        if mlp1_results[i] == "true" and clf_results[i] == "false":
            count_shared += 1
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "false" and tree_results[i] == "true":
        if mlp1_results[i] == "false" and clf_results[i] == "true":
            count_shared += 1
for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "false" and tree_results[i] == "true":
        if mlp1_results[i] == "true" and clf_results[i] == "true":
            count_shared += 1

for i in range(len(mlp_results) - 1):
    if mlp_results[i] == "false" and tree_results[i] == "false":
        if mlp1_results[i] == "true" and clf_results[i] == "true":
            count_shared += 1

print("Percentage where 2 or 3 were correct:")
print(count_shared/len(mlp_results))

weight_mlp = 1
weight_tree = 1

for i in range (len(y_pred)):
    if y_pred[i] == Y_test[i]:
        count += 1
    else:
        weight_mlp = weight_mlp/2

for i in range (len(y_pred_tree)):
    if y_pred_tree[i] == Y_test[i]:
        count_tree += 1
    else:
        weight_tree = weight_tree/2



for i in range (len(y_pred)):
    if y_pred[i] == Y_test[i]:
        mlp_results.append("true")
    else: 
        mlp_results.append("false")

for i in range (len(y_pred_tree)):
    if y_pred_tree[i] == Y_test[i]:
        tree_results.append("true")
    else: 
        tree_results.append("false")

for i in range (len(y_pred_clf)):
    if y_pred_clf[i] == Y_test[i]:
        clf_results.append("true")
    else: 
        clf_results.append("false")



one_weight = 0
two_weight = 0
three_weight = 0
four_weight = 0
five_weight = 0

weighted_results = []
mlp_weight = 1
mlp1_weight = 1
tree_weight = 1
clf_weight = 1

for i in range (len(y_pred) - 1):
    if mlp_results[i] == "true":
        mlp_weight = mlp_weight * 2
    else:
        mlp_weight = mlp_weight / 2
    if mlp1_results[i] == "true":
        mlp1_weight = mlp1_weight * 2
    else:
        mlp1_weight = mlp1_weight / 2
    if tree_results[i] == "true":
        tree_weight = tree_weight * 2
    else:
        tree_weight = tree_weight / 2
    if clf_results[i] == "true":
        clf_weight = clf_weight * 2
    else:
        clf_weight = clf_weight / 2
    

    if y_pred[i] == "1":
        one_weight = one_weight + mlp_weight
    if y_pred[i] == "2":
        two_weight = two_weight + mlp_weight
    if y_pred[i] == "3":
        three_weight = three_weight + mlp_weight
    if y_pred[i] == "4":
        four_weight = four_weight + mlp_weight
    if y_pred[i] == "5":
        five_weight = five_weight + mlp_weight

    if y_pred1[i] == "1":
        one_weight = one_weight + mlp1_weight
    if y_pred1[i] == "2":
        two_weight = two_weight + mlp1_weight
    if y_pred1[i] == "3":
        three_weight = three_weight + mlp1_weight
    if y_pred1[i] == "4":
        four_weight = four_weight + mlp1_weight
    if y_pred1[i] == "5":
        five_weight = five_weight + mlp1_weight

    if y_pred_tree[i] == "1":
        one_weight = one_weight + tree_weight
    if y_pred_tree[i] == "2":
        two_weight = two_weight + tree_weight
    if y_pred_tree[i] == "3":
        three_weight = three_weight + tree_weight
    if y_pred_tree[i] == "4":
        four_weight = four_weight + tree_weight
    if y_pred_tree[i] == "5":
        five_weight = five_weight + tree_weight

    if y_pred_clf[i] == "1":
        one_weight = one_weight + clf_weight
    if y_pred_clf[i] == "2":
        two_weight = two_weight + clf_weight
    if y_pred_clf[i] == "3":
        three_weight = three_weight + clf_weight
    if y_pred_clf[i] == "4":
        four_weight = four_weight + clf_weight
    if y_pred_clf[i] == "5":
        five_weight = five_weight + clf_weight


    if one_weight >= two_weight:
        if one_weight >= three_weight:
            if one_weight >= four_weight:
                if one_weight >= five_weight:
                    weighted_results.append(1)
    if two_weight >= one_weight:
        if two_weight >= three_weight:
            if two_weight >= four_weight:
                if two_weight >= five_weight:
                    weighted_results.append(2)
    if three_weight >= two_weight:
        if three_weight >= one_weight:
            if three_weight >= four_weight:
                if three_weight >= five_weight:
                    weighted_results.append(3)
    if four_weight >= two_weight:
        if four_weight >= three_weight:
            if four_weight >= one_weight:
                if four_weight >= five_weight:
                    weighted_results.append(4)
    if five_weight >= two_weight:
        if five_weight >= three_weight:
            if five_weight >= four_weight:
                if five_weight >= five_weight:
                    weighted_results.append(5)
        


count_weighted = 0

for i in range (len(Y_test) - 1):
    if weighted_results[i] == Y_test[i]:
        count_weighted += 1
    else:
        if i < 20:
            print(weighted_results[i])
            print(Y_test[i])
 
#debug statements
print(count_weighted)
print(len(Y_test))

print(count_weighted/len(Y_test))

