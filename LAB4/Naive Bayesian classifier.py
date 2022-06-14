import pandas as pd

data = pd.read_csv('Desktop/dat4.csv.csv')
data.head()
y = list(data['PlayTennis'].values)
X = data.iloc[:,1:].values

print(f'Target Values: {y}')
print(f'Features: \n{X}')
y_train = y[:8]
y_val = y[8:]

X_train = X[:8]
X_val = X[8:]

print(f"Number of instances in training set: {len(X_train)}")
print(f"Number of instances in testing set: {len(X_val)}")
class NaiveBayesClassifier:
    
    
    def __init__(self, X, y):
        
        self.X, self.y = X, y 
        
        self.N = len(self.X)

        self.dim = len(self.X[0]) 

        self.attrs = [[] for _ in range(self.dim)] 

        self.output_dom = {} 

        self.data = []
        
        for i in range(len(self.X)):
            for j in range(self.dim):
                if not self.X[i][j] in self.attrs[j]:
                    self.attrs[j].append(self.X[i][j])
                    
            if not self.y[i] in self.output_dom.keys():
                self.output_dom[self.y[i]] = 1
            
            else:
                self.output_dom[self.y[i]] += 1
            
            self.data.append([self.X[i], self.y[i]])
    def classify(self, entry):

        solve = None 
        max_arg = -1

        for y in self.output_dom.keys():

            prob = self.output_dom[y]/self.N 

            for i in range(self.dim):
                cases = [x for x in self.data if x[0][i] == entry[i] and x[1] == y] 
                n = len(cases)
                prob *= n/self.N 
                
            if prob > max_arg:
                max_arg = prob
                solve = y

        return solve
nbc = NaiveBayesClassifier(X_train, y_train)

total_cases = len(y_val)

good = 0
bad = 0
predictions = []

for i in range(total_cases):
    predict = nbc.classify(X_val[i])
    predictions.append(predict)

    if y_val[i] == predict:
        good += 1
    else:
        bad += 1
        
print('Predicted values:', predictions)
print('Actual values:', y_val)
print()
print('Total number of testing instances in the dataset:', total_cases)
print('Number of correct predictions:', good)
print('Number of wrong predictions:', bad)
print()
print('Accuracy of Bayes Classifier:', good/total_cases)
