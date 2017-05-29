'''
contains the model related methods
'''
import DataLoader
import Util
import numpy as np
import pandas as pd

def changeContact(series):
    if series['contact']=='telephone':
        return 1
    elif series['contact']== 'cellular':
        return 0
    else:
        return -1

def updatePoutcome(df):
    if df['poutcome'] == 'nonexistent': return 1
    elif df['poutcome'] == 'failure': return 2
    else:
        return 3

def updateMarital(df):
    #'married' 'divorced' 'single' 'unknown']
    if df['marital'] == 'single': return 1
    elif df['marital'] == 'married': return 2
    elif df['marital'] == 'divorced':
        return 3
    else:
        return 4

def updateProfession(df):
    '''
    'self-employed' 'housemaid' 'management' 'blue-collar' 'technician'
 'admin.' 'entrepreneur' 'retired' 'services' 'unemployed' 'student'
 'unknown'
    :param df:
    :return:
    '''
    if df['profession'] == 'self-employed': return 1
    elif df['profession'] == 'housemaid': return 2
    elif df['profession'] == 'management':
        return 3
    elif df['profession'] == 'blue-collar':
        return 4
    elif df['profession'] == 'technician':
        return 5
    elif df['profession'] == 'admin.':
        return 6
    elif df['profession'] == 'entrepreneur':
        return 7
    elif df['profession'] == 'retired':
        return 8
    elif df['profession'] == 'services':
        return 9
    elif df['profession'] == 'unemployed':
        return 10
    elif df['profession'] == 'student':
        return 11
    else:
        return 12


def updateSchooling(df):
    #['basic.9y' 'professional.course' 'university.degree' 'basic.4y' nan
    #'high.school' 'basic.6y' 'unknown' 'illiterate']
    if df['schooling'] == 'illiterate': return 1
    elif df['schooling'] == 'basic.4y': return 2
    elif df['schooling'] == 'basic.6y':
        return 3
    elif df['schooling'] == 'basic.9y':
        return 4
    elif df['schooling'] == 'high.school':
        return 5
    elif df['schooling'] == 'professional.course':
        return 6
    elif df['schooling'] == 'university.degree':
        return 7
    else:
        return 8 # for unknown and nan

def updateDayofWeek(df):
    if df['day_of_week'] == 'sun': return 1
    elif df['day_of_week'] == 'mon': return 2
    elif df['day_of_week'] == 'mon':
        return 3
    elif df['day_of_week'] == 'mon':
        return 4
    elif df['day_of_week'] == 'mon':
        return 5
    elif df['day_of_week'] == 'mon':
        return 6
    elif df['day_of_week'] == 'mon':
        return 7
    else:
        return 1 #default sunday

def updateMonth(df):
    if df['month'] == 'jan': return 1
    elif df['month'] == 'feb': return 2
    elif df['month'] == 'mar':
        return 3
    elif df['month'] == 'apr':
        return 4
    elif df['month'] == 'may':
        return 5
    elif df['month'] == 'jun':
        return 6
    elif df['month'] == 'jul':
        return 7
    elif df['month'] == 'aug':
        return 8
    elif df['month'] == 'sep':
        return 9
    elif df['month'] == 'oct':
        return 10
    elif df['month'] == 'nov':
        return 11
    elif df['month'] == 'dec':
        return 12
    else:
        return 1

'''
given the dataframe,
analyze the fields for missing values,
excludes the fields that are not relevant
'''
def preprocessData(trainData):
    #lets see the columns
    #print(trainData.columns.values)
    '''
    ['custAge' 'profession' 'marital' 'schooling' 'default' 'housing' 'loan'
 'contact' 'month' 'day_of_week' 'campaign' 'pdays' 'previous' 'poutcome'
 'emp.var.rate' 'cons.price.idx' 'cons.conf.idx' 'euribor3m' 'nr.employed'
 'pmonths' 'pastEmail' 'responded' 'profit' 'id']
    '''
    #id is just a serial number so it can be excluded in modeling
    #profit is not useful
    #check if profit is present (as this method is done for test data which does not have profit column)
    fields = np.array(trainData.columns.values)
    if 'id' in fields:
        trainData = trainData.drop(['id'], axis = 1)
    if 'profit' in fields:
        trainData = trainData.drop(['profit'], axis=1)
    #print(trainData.columns.values)

    #convert the fields into numeric
    #housing, loan, responded yes =1, no = 0
    if 'responded' in fields:
        trainData.loc[trainData['responded'] == 'yes', 'responded_new'] = 1
        trainData.loc[trainData['responded'] == 'no', 'responded_new'] = 0
        # for fields with other values, assign 1 or 0, based on the max occurences
        respondedYesCount = trainData[trainData['responded_new'] == 1].shape[0]
        respondedNoCount = trainData[trainData['responded_new'] == 0].shape[0]

        trainData.loc[(trainData['responded'] != 'no') & (
        trainData['responded'] != 'yes'), 'responded_new'] = 1 if respondedYesCount > respondedNoCount else 0

    trainData.loc[trainData['housing'] == 'yes', 'housing_new'] = 1
    trainData.loc[trainData['housing'] == 'no', 'housing_new'] = 0
    #for the other types, simply assign whichever is maximum
    # for fields with other values, assign 1 or 0, based on the max occurences
    housingYesCount = trainData[trainData['housing_new']==1].shape[0]
    housingNoCount = trainData[trainData['housing_new'] == 0].shape[0]

    trainData.loc[(trainData['housing'] != 'no') & (trainData['housing']!='yes'), 'housing_new'] = 1 if housingYesCount>housingNoCount else 0

    trainData.loc[trainData['loan'] == 'yes', 'loan_new'] = 1
    trainData.loc[trainData['loan'] == 'no', 'loan_new'] = 0
    loanYesCount = trainData[trainData['housing_new'] == 1].shape[0]
    loanNoCount = trainData[trainData['housing_new'] == 0].shape[0]
    #for fields with other values, assign 1 or 0, based on the max occurences
    trainData.loc[(trainData['loan'] != 'no') & (trainData[
        'loan'] != 'yes'), 'loan_new'] = 1 if loanYesCount > loanNoCount else 0

    #print(trainData['contact'].unique())
    # contact has just two type, cellular = 1, telephone = 2
    trainData.loc[trainData['contact'] == 'telephone', 'contact_new'] = 1
    trainData.loc[trainData['contact'] == 'cellular', 'contact_new'] = 0


    #change the marital field
    #lets see the marital field's values
    #print(trainData['marital'].unique())
    #['married' 'divorced' 'single' 'unknown']
    trainData['marital_new'] = trainData.apply(updateMarital, axis=1)


    #change the schooling field
    '''
    ['basic.9y' 'professional.course' 'university.degree' 'basic.4y' nan
 'high.school' 'basic.6y' 'unknown' 'illiterate']
    '''
    #print(trainData['schooling'].unique())
    trainData['schooling_new'] = trainData.apply(updateSchooling, axis=1)

    #print(trainData['default'].unique())
    #['no' 'unknown' 'yes']
    trainData.loc[trainData['default'] == 'yes', 'default_new'] = 1
    trainData.loc[trainData['default'] == 'no', 'default_new'] = 0
    # for the other types, simply assign whichever is maximum
    # for fields with other values, assign 1 or 0, based on the max occurences
    defaultYesCount = trainData[trainData['default_new'] == 1].shape[0]
    defaultNoCount = trainData[trainData['default_new'] == 0].shape[0]

    trainData.loc[(trainData['default'] != 'no') & (
    trainData['default'] != 'yes'), 'default_new'] = 1 if defaultYesCount > defaultNoCount else 0

    #check the profession field
    #print(trainData['profession'].unique())
    '''
    ['self-employed' 'housemaid' 'management' 'blue-collar' 'technician'
 'admin.' 'entrepreneur' 'retired' 'services' 'unemployed' 'student'
 'unknown']
    '''
    trainData['profession_new'] = trainData.apply(updateProfession, axis=1)


    #now change the months
    trainData['month_new'] = trainData.apply(updateMonth, axis = 1)

    #change the day of week
    trainData['day_of_week_new'] = trainData.apply(updateDayofWeek, axis=1)

    #lets see the types of poutcome
    #print(trainData['poutcome'].unique())
    #['nonexistent' 'failure' 'success']
    # change the poutcome filed
    trainData['poutcome_new'] = trainData.apply(updatePoutcome, axis=1)


    #print(trainData['poutcome'],trainData['poutcome_new'])

    #we just retain the new fields or those which are numeric and drop others
    trainData = trainData.drop(['housing', 'loan', 'contact', 'month',
                                'day_of_week', 'poutcome',
                                'marital', 'schooling', 'default', 'profession'], axis=1)
    if 'responded' in fields: # this is required because the test data doesnot contain 'responded' column
        trainData = trainData.drop(['responded'], axis =1)

    #print the columns we will be using
    #print(trainData.columns.values)
    '''
    ['custAge' 'profession' 'marital' 'schooling' 'default' 'campaign' 'pdays'
 'previous' 'emp.var.rate' 'cons.price.idx' 'cons.conf.idx' 'euribor3m'
 'nr.employed' 'pmonths' 'pastEmail' 'responded_new' 'housing_new'
 'loan_new' 'contact_new' 'month_new' 'day_of_week_new' 'poutcome_new']
    '''

    #the custage field has many NA values, we assign it with the age that occurs frequently
    ageCounts = trainData.groupby('custAge').size().nlargest(1).reset_index(name='top1')

    maxAgeCount = ageCounts['custAge'][0]
    #we can fill with max or with mean
    trainData.custAge.fillna(maxAgeCount, inplace=True)

    return trainData


'''
find the accuracy of the predictions
'''
def accuracy(y_true, y_pred):
    correct_pred = 0
    for index,item in enumerate(y_pred):
        if item == y_true[index]:
            correct_pred+=1
    #print("correct predictions",correct_pred)
    accuracy = correct_pred / float(len(y_true))
    return accuracy


def predictUsingRandomForest(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)
    accuracyScore = accuracy(y_test, y_pred)
    print("Randomforest score on training set: ", accuracyScore)
    return random_forest, accuracyScore

def predictUsingNaiveBayes(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    y_pred = gaussian.predict(X_test)
    accuracyScore = accuracy(y_test, y_pred)
    print("NaiveBayes score on training set: ", accuracyScore)
    return gaussian, accuracyScore


'''
predicts using SVM
'''
def predictUsingSVM(X_train, y_train, X_test, y_test):
    # now train the model
    from sklearn.svm import SVC
    myModel = SVC()
    myModel.fit(X_train, y_train)
    # check with the test data and see the performance of the model
    y_pred = myModel.predict(X_test)

    # compare the predicted vs actual value
    accuracyScore = accuracy(y_test, y_pred)
    print("SVM score on training set: ", accuracyScore)
    return myModel,accuracyScore

def predictUsingLasso(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import Lasso

    # find the alpha through cross-validation.
    best_alpha = 0.00099

    myModel = Lasso(alpha=best_alpha, max_iter=500000, normalize=True)
    myModel.fit(X_train, y_train)

    # predict on test data
    y_pred = myModel.predict(X_test)
    accuracyScore = accuracy(y_test, y_pred)
    print("Lasso score on training set: ", accuracyScore)
    return myModel, accuracyScore

def predictUsingLogisticRegression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    myModel = LogisticRegression()
    myModel.fit(X_train, y_train)
    y_pred = myModel.predict(X_test)
    accuracyScore = accuracy(y_test, y_pred)
    print("Logistic regression score on training set: ", accuracyScore)
    return myModel, accuracyScore

def predictUsingGradientDescent(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import SGDClassifier

    myModel = SGDClassifier(loss="hinge", penalty="l2")
    myModel.fit(X_train, y_train)
    predictions = []
    for item in X_test:
        predictions.append(myModel.predict([item])[0])
    # compare the predicted vs actual value
    accuracyScore = accuracy(y_test, predictions)
    print("Gradient Descent regression score on training set: ", accuracyScore)
    return myModel, accuracyScore


def predictusingKNN(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracyScore = accuracy(y_test, y_pred)
    print("KNN score on training set: ", accuracyScore)
    return knn, accuracyScore


'''
trains a basic model using SVM
'''
def trainModel(trainData):
    '''
    trains the model using SVM, the target variable is responded_new
    and the other variables are used to predict the target variable
    :param trainData: the train dataframe
    :return: the trained model
    '''

    #Using all the fields, we get the following result from the two models

    '''
    ('correct predictions', 1111)
    ('SVM score on training set: ', 0.8988673139158576)
    ('correct predictions', 1121)
    ('Random forest score on training set: ', 0.906957928802589)
    ('correct predictions', 1102)
    ('KNN score on training set: ', 0.8915857605177994)
    ('correct predictions', 1122)
    ('Gradient Descent regression score on training set: ', 0.9077669902912622)
    ('correct predictions', 1125)
    ('Logistic regression score on training set: ', 0.9101941747572816)
    ('correct predictions', 1017)
    ('NaiveBayes score on training set: ', 0.8228155339805825)
    '''
    #only use the fields that have some impact (got this  information from data analysis)
    '''
    X = np.array(trainData[['custAge', 'profession_new', 'marital_new', 'schooling_new',
                            'default_new', 'campaign', 'pdays',
        'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
        'nr.employed', 'pmonths', 'pastEmail', 'housing_new',
        'loan_new', 'contact_new', 'month_new', 'day_of_week_new', 'poutcome_new']])
    '''
    X = np.array(
        trainData[['custAge', 'profession_new', 'marital_new',
                   'schooling_new', 'default_new','pdays','previous', 'emp.var.rate', 'cons.price.idx',
                   'cons.conf.idx', 'euribor3m','pmonths','loan_new', 'contact_new', 'month_new','day_of_week_new'
                   ,'housing_new'
                   ]])

    #with these set of fields, we have following performance
    '''
    ('SVM score on training set: ', 0.9021035598705501)
    ('Randomforest score on training set: ', 0.8932038834951457)
    ('KNN score on training set: ', 0.8964401294498382)
    ('Gradient Descent regression score on training set: ', 0.9077669902912622)
    ('Logistic regression score on training set: ', 0.9093851132686084)
    ('NaiveBayes score on training set: ', 0.8762135922330098)
    '''
    y = np.array(trainData['responded_new'])#np.array([1, 1, 2, 2])

    # we would like to validate the model based on the training and validation set
    #split the data into train and validation set and see the performance of the model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

    svmModel, svmscore = predictUsingSVM(X_train, y_train, X_test, y_test)

    regModel, regscore = predictUsingRandomForest(X_train, y_train, X_test, y_test)

    knnModel, knnScore = predictusingKNN(X_train, y_train, X_test, y_test)

    gdModel, gdScore = predictUsingGradientDescent(X_train, y_train, X_test, y_test)

    lregModel, lregScore = predictUsingLogisticRegression(X_train, y_train, X_test, y_test)

    naiveModel, naiveScore = predictUsingNaiveBayes(X_train, y_train, X_test, y_test)
    #use the best model for prediction on test data
    models = ["svm", "rforest", "knn", "sgd" ,"lreg", "nb"]
    scores = [svmscore, regscore, knnScore, gdScore, lregScore, naiveScore]
    print("models and scores:")
    print(models)
    print(scores)
    bestScore = np.max(np.array(scores))
    bestModelIndex = np.where(scores==bestScore)[0][0]
    print("best model is:",models[bestModelIndex])

    if bestModelIndex==0:
        myModel = svmModel
    elif bestModelIndex==1:
        myModel = regModel
    elif bestModelIndex==2:
        myModel = knnModel
    elif bestModelIndex==3:
        myModel = gdModel
    elif bestModelIndex==4:
        myModel = lregModel
    elif bestModelIndex==5:
        myModel = naiveModel
    else:
        myModel = svmModel

    return myModel

'''
for every row, does the label prediction
'''
def predictLabel(model, testData):
    '''

    :param model: the trained model
    :param testData: the test data
    :return:
    '''
    #lets see the fields in the test data
    #print(testData.columns.values)
    '''
    ['custAge' 'campaign' 'pdays' 'previous' 'emp.var.rate' 'cons.price.idx'
     'cons.conf.idx' 'euribor3m' 'nr.employed' 'pmonths' 'pastEmail'
     'housing_new' 'loan_new' 'contact_new' 'marital_new' 'schooling_new'
     'default_new' 'profession_new' 'month_new' 'day_of_week_new'
     'poutcome_new']
    '''
    #for every row, we predict the label
    X = np.array(
        testData[['custAge', 'profession_new', 'marital_new',
                   'schooling_new', 'default_new', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
                   'cons.conf.idx', 'euribor3m', 'pmonths', 'loan_new', 'contact_new', 'month_new',
                  'day_of_week_new'
                  ,'housing_new']])
    y_pred = model.predict(X)
    testData['market_flag'] = y_pred
    return testData

'''
performs the analysis on age field
'''
def analyzeAgeEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    df = trainData[['custAge', 'responded_new']]
    df['custAge'].hist(by=df['responded_new'])
    plt.show()


'''
performs the analysis on age field
'''
def analyzeSchoolingEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    df = trainData[['schooling_new', 'responded_new']]
    df['responded_new'].hist(by=df['schooling_new'])
    plt.show()

'''
performs the analysis on age field
'''
def analyzeProfessionEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    df = trainData[['profession_new', 'responded_new']]
    df['responded_new'].hist(by=df['profession_new'])
    plt.show()

def analyzecampaignEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    df = trainData[['campaign', 'responded_new']]
    df['campaign'].hist(by=df['responded_new'], figsize = (15,15))
    plt.show()

def analyzepDaysEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    #check the unique value of pdays
    print(trainData['pdays'].unique())
    df = trainData[['pdays', 'responded_new']]
    df['pdays'].hist(by=df['responded_new'], figsize=(15, 15))
    plt.show()

def analyzePreviousEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['previous'].unique())
    df = trainData[['previous', 'responded_new']]
    df['responded_new'].hist(by=df['previous'], figsize=(15, 15))
    plt.show()

def analyzevarRateEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['emp.var.rate'].unique())
    df = trainData[['emp.var.rate', 'responded_new']]
    df['responded_new'].hist(by=df['emp.var.rate'], figsize=(15, 15))
    plt.show()

def analyzeConsPriceEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['cons.price.idx'].unique())
    df = trainData[['cons.price.idx', 'responded_new']]
    df['cons.price.idx'].hist(by=df['responded_new'], figsize=(75, 75))
    plt.show()

def analyzeconsConfEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['cons.conf.idx'].unique())
    df = trainData[['cons.conf.idx', 'responded_new']]
    df['cons.conf.idx'].hist(by=df['responded_new'], figsize=(15, 15))
    plt.show()

def analyzeEuriborEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['euribor3m'].unique())
    df = trainData[['euribor3m', 'responded_new']]
    df['euribor3m'].hist(by=df['responded_new'], figsize=(15, 15))
    plt.show()

def analyzePMonthsEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['pmonths'].unique())
    df = trainData[['pmonths', 'responded_new']]
    df['pmonths'].hist(by=df['responded_new'], figsize=(15, 15))
    plt.show()

def analyzeEmployedEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['nr.employed'].unique())
    df = trainData[['nr.employed', 'responded_new']]
    df['responded_new'].hist(by=df['nr.employed'], figsize=(15, 15))
    plt.show()


def analyzePastEmailEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['pastEmail'].unique())
    df = trainData[['pastEmail', 'responded_new']]
    df['responded_new'].hist(by=df['pastEmail'], figsize=(15, 15))
    plt.show()


def analyzeHousingEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['housing_new'].unique())
    df = trainData[['housing_new', 'responded_new']]
    df['responded_new'].hist(by=df['housing_new'], figsize=(15, 15))
    plt.show()

def analyzeLoanEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['loan_new'].unique())
    df = trainData[['loan_new', 'responded_new']]
    df['responded_new'].hist(by=df['loan_new'], figsize=(15, 15))
    plt.show()


def analyzeContactEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['contact_new'].unique())
    df = trainData[['contact_new', 'responded_new']]
    df['responded_new'].hist(by=df['contact_new'], figsize=(15, 15))
    plt.show()


def analyzeMaritalEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['marital_new'].unique())
    df = trainData[['marital_new', 'responded_new']]
    df['responded_new'].hist(by=df['marital_new'], figsize=(15, 15))
    plt.show()


def analyzeDefaultEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['default_new'].unique())
    df = trainData[['default_new', 'responded_new']]
    df['default_new'].hist(by=df['responded_new'], figsize=(55, 55))
    plt.show()


def analyzeMonthEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['month_new'].unique())
    df = trainData[['month_new', 'responded_new']]
    df['responded_new'].hist(by=df['month_new'], figsize=(15, 15))
    plt.show()


def analyzeMonthEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['month_new'].unique())
    df = trainData[['month_new', 'responded_new']]
    df['responded_new'].hist(by=df['month_new'], figsize=(15, 15))
    plt.show()

def analyzePoutcomeEffect(trainData):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    # check the unique value of pdays
    print(trainData['poutcome_new'].unique())
    df = trainData[['poutcome_new', 'responded_new']]
    df['responded_new'].hist(by=df['poutcome_new'], figsize=(15, 15))
    plt.show()

'''
does the graphical analysis of the effect of fields in the target label
'''
def doGraphicalAnalysis(trainData):
    print("doing graphical analysis")
    '''
    ['custAge' 'campaign' 'pdays' 'previous' 'emp.var.rate' 'cons.price.idx'
     'cons.conf.idx' 'euribor3m' 'nr.employed' 'pmonths' 'pastEmail'
     'housing_new' 'loan_new' 'contact_new' 'marital_new' 'schooling_new'
     'default_new' 'profession_new' 'month_new' 'day_of_week_new'
     'poutcome_new']
    :return:
    '''

    #analyzeProfessionEffect(trainData)

    #analyzeSchoolingEffect(trainData)

    #analyzeAgeEffect(trainData)

    #analyzecampaignEffect(trainData)

    #analyzepDaysEffect(trainData)

    #analyzePreviousEffect(trainData)

    #analyzevarRateEffect(trainData)


    #analyzeConsPriceEffect(trainData)

    #analyzeconsConfEffect(trainData)

    #analyzeEuriborEffect(trainData)

    #analyzeEmployedEffect(trainData)
    #analyzePastEmailEffect(trainData)
    #analyzePMonthsEffect(trainData)

    #analyzeHousingEffect(trainData)

    #analyzeLoanEffect(trainData)
    #analyzeContactEffect(trainData)

    #analyzeMaritalEffect(trainData)
    #analyzeDefaultEffect(trainData)

    #analyzeMonthEffect(trainData)
    #analyzepDaysEffect(trainData)

    #analyzePoutcomeEffect(trainData)


if __name__=="__main__":
    print(" inside main method")
    trainData = DataLoader.loadData(Util.trainFilePath)
    trainData = preprocessData(trainData)
    trainData.to_csv('traindata_processed.csv')
    #we donot know if the given fields are even useful for prediction or not, so we graphically analyze this
    doGraphicalAnalysis(trainData)
    '''
    **************
    from the plots, we can see the following observations:
    ***************
    - profession has some correlation with response
    - schooling seems to have very less correlation with response
    -consprice has effect
    -age seems to have some effect
    -consconf has some effect
    -pdays has some effect
    -varrate has some effect
    -euribor has some effect
    -pmonths has some effect
    -loan has some effect
    -contact has some effect
    -default has some effect
    -marital has less effect
    -month has some effect
    -days has some effect
    ***********************
    -poutcome has very less effect
    -housing has very less effect
    -emplyed has very less effect
    -past email has very less effect
    -campaign has very less effect
    *************************
    '''

    model = trainModel(trainData)

    testData = DataLoader.loadData(Util.testFilePath)
    testData = preprocessData(testData)
    testData.to_csv('testdata_processed.csv')
    #now we do the prediction
    testData = predictLabel(model, testData)
    testData.to_csv(Util.resultFilePath)


