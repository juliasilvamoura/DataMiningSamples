# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

def load_dataset(input_file): 
    names = ['Clump-Thickness','Cell-Size','Cell-Shape','Marginal-Adhesion','Single-Epithelial-Cell-Size','Bare-Nuclei','Bland-Chromatin','Normal-Nucleoli','Mitoses','Class']
    features = ['Clump-Thickness','Cell-Size','Cell-Shape','Marginal-Adhesion','Single-Epithelial-Cell-Size','Bare-Nuclei','Bland-Chromatin','Normal-Nucleoli', 'Mitoses','Class']
    target = 'Class'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas  
    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:,[target]].values   

    return x,y       
    


def main():
    input_file = 'Datasets/breast-cancer-output.data'
    names = ['Clump-Thickness','Cell-Size','Cell-Shape','Marginal-Adhesion','Single-Epithelial-Cell-Size','Bare-Nuclei','Bland-Chromatin','Normal-Nucleoli','Mitoses','Class']
    features = ['Clump-Thickness','Cell-Size','Cell-Shape','Marginal-Adhesion','Single-Epithelial-Cell-Size','Bare-Nuclei','Bland-Chromatin','Normal-Nucleoli', 'Mitoses','Class']
    target = 'Class'
    data = pd.read_csv(input_file, names=names)
    df = pd.DataFrame(data=data, columns=data.columns)
    df['target'] = data['Class']
    print(df.head())
   
   
    #load dataset
    #target_names, df = load_dataset('iris')

    # Separate X and y data
    x = df.drop('target', axis=1)
    y = df.target   

    print("Total samples: {}".format(x.shape[0]))

    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # TESTS USING SVM classifier from sk-learn    
    svm = SVC(kernel='poly') # poly, rbf, linear
    # training using train dataset
    svm.fit(X_train, y_train)
    # get support vectors
    print(svm.support_vectors_)
    # get indices of support vectors
    print(svm.support_)
    # get number of support vectors for each class
    print("Qtd Support vectors: ")
    print(svm.n_support_)
    # predict using test dataset
    y_hat_test = svm.predict(X_test)

     # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test,average='macro')
    print("Acurracy SVM from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.2f}%".format(f1))

    # Get test confusion matrix    
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, ['nao cancer', 'cancer'], False, "Confusion Matrix - SVM sklearn")      
    plot_confusion_matrix(cm, ['nao cancer', 'cancer'], True, "Confusion Matrix - SVM sklearn normalized" )  

    cross_validation = cross_val_score(svm, x, y, cv=10)
    print("Cross Validation 10")
    print(cross_validation)
    print("Cross Validation Média")
    print(cross_validation.mean()) #media do cross validation 
    plt.show()


if __name__ == "__main__":
    main()