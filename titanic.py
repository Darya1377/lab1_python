import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import argparse
#import xgboost


def prepare_num(df):
    """

    Parameters
    ----------
    df = input dataframe

    Returns
    -------
    df_num = improved dataframe
    """
    df_num = df.drop(['age_child', 'age_adult', 'age_old', 'morning', 'day', 'evening'], axis=1)
    df_age1 = pd.get_dummies(df['age_child'], prefix='age_child')
    df_age2 = pd.get_dummies(df['age_adult'], prefix='age_adult')
    df_age3 = pd.get_dummies(df['age_old'], prefix='age_old')
    df_age = pd.get_dummies(df['morning'], prefix='morning')
    df_pcl = pd.get_dummies(df['day'], prefix='day')
    df_pcl2 = pd.get_dummies(df['evening'], prefix='evening')

    df_num = pd.concat((df_num, df_age1, df_age2, df_age3, df_age, df_pcl, df_pcl2), axis=1)
    return df_num


def accuracy(model, df_prep_x_num_tst, df_prep_y_tst):
    """

    Parameters
    ----------
    model = input model
    df_prep_x_num_tst = train dataframe
    df_prep_y_tst = predictable label

    Returns
    -------
    accuracy of a model
    """
    return model.score(df_prep_x_num_tst, df_prep_y_tst)


def decisionTree(df_prep_x_num, df_prep_y):
    """

    Parameters
    ----------
    df_prep_x_num = main dataframe
    df_prep_y = main predictable label

    Returns
    -------
    model_tree = decisionTree model
    """
    model_tree = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    model_tree = model_tree.fit(df_prep_x_num, df_prep_y)

    return model_tree


def decision_tree_out(model_tree,df_prep_x_num):
    """

    Parameters
    ----------
    model_tree = input model
    df_prep_x_num = main dataframe

    Returns
    -------
    shows model tree
    """
    tree.plot_tree(model_tree, feature_names=df_prep_x_num.columns, filled=True)
    plt.show()
    return()


def decision_tree_with_2_features(df_prep_x_num,df_prep_y):
    """

    Parameters
    ----------
    df_prep_x_num = main dataframe
    df_prep_y = main predictable label

    Returns
    -------
    m_DT2 = model of a decision tree with 2 features
    """
    df_trainx_2 = df_prep_x_num.drop(['sex', 'row_number', 'liters_drunk', 'drink', 'check_number', 'age_child_False',
                                      'age_child_True'], axis=1)
    df_trainxx_2 = df_trainx_2.drop(
        ['age_adult_True', 'age_adult_False', 'age_old_True', 'age_old_False', 'morning_True',
         'day_True', 'evening_False', 'evening_True'], axis=1)
    m_DT2 = decisionTree(df_trainxx_2, df_prep_y)
    return m_DT2


def test_drop(df_prep_x_num_tst):
    """

    Parameters
    ----------
    df_prep_x_num_tst = train dataframe

    Returns
    -------
    df_trainxx_2 = improved train dataframe
    """
    df_trainx_2 = df_prep_x_num_tst.drop(['sex', 'row_number', 'liters_drunk', 'drink', 'check_number', 'age_child_False',
                                          'age_child_True'], axis=1)
    df_trainxx_2 = df_trainx_2.drop(
        ['age_adult_True', 'age_adult_False', 'age_old_True', 'age_old_False', 'morning_True',
         'day_True', 'evening_False', 'evening_True'], axis=1)
    return df_trainxx_2


'''''
def xgboost(df_prep_x_num, df_prep_y, df_prep_x_num_tst):
    model_xgboost = XGBClassifier(n_estimators=20, max_depth=4)
    model_xgboost.fit(df_prep_x_num, df_prep_y)

    predict = model_xgboost.predict(df_prep_x_num_tst)
    return predict
    
 '''

def logisticRegression():
    """

    Returns
    -------
    model_LogR = model of Logistic Regression
    """
    model_LogR = LogisticRegression(C=0.1, solver='lbfgs')
    return model_LogR


def main():

    parser = argparse.ArgumentParser(description='Different methods of machine learning')
    parser.add_argument("--model_type",
                        choices=["xgboost", "Decision-Tree", "Decision-Tree-with-2-main-features",
                                 "Logistic-Regression"],
                        required=True, type=str, help="Type in model type")
    args = parser.parse_args()
    m = args.model_type

    df_main = pd.read_csv('titanic_prepared.csv')
    q = round(len(df_main.index) * 0.1)

    df_test = df_main.sample(n=q)

    df_prep_x = df_main.drop(['label'], axis = 1)
    df_prep_x_tst = df_test.drop(['label'], axis = 1)
    df_prep_y = df_main['label']
    df_prep_y_tst = df_test['label']

    df_prep_x_num = prepare_num(df_prep_x)
    df_prep_x_num_tst = prepare_num(df_prep_x_tst)
    df_prep_x_num = df_prep_x_num.fillna(df_prep_x_num.median())
    df_prep_x_num_tst = df_prep_x_num_tst.fillna(df_prep_x_num.median())

    m_DT = decisionTree(df_prep_x_num, df_prep_y)
    m_DT2 = decision_tree_with_2_features(df_prep_x_num, df_prep_y)
    m_LogR = logisticRegression()
    m_DT2 = decision_tree_with_2_features(df_prep_x_num, df_prep_y)
    # m_XGB = xgboost(df_prep_x_num, df_prep_y, df_prep_x_num_tst)

    if m == "Decision-Tree":
        mod_DT = decision_tree_out(m_DT, df_prep_x_num)
        print("Accuracy of DecisionTree", m_DT.score(df_prep_x_num_tst, df_prep_y_tst))
    if m == "Decision-Tree-with-2-main-features":
        mod_DT2 = decision_tree_out(m_DT2, df_prep_x_num)
        print("Accuracy of DecisionTree with 2 main features", accuracy(m_DT2, test_drop(df_prep_x_num_tst),
                                                                        df_prep_y_tst))
    if m == "Logistic-Regression":
        mod_LogR = m_LogR.fit(df_prep_x_num, df_prep_y)
        y = mod_LogR.predict(df_prep_x_num_tst)
        print(y)
        print("Accuracy of Logistic Regression", accuracy(mod_LogR, df_prep_x_num_tst, df_prep_y_tst))

''''  
    if m == "xgboost":
        mod_LogR = m_LogR.fit(df_prep_x_num, df_prep_y)
        y = mod_LogR.predict(df_prep_x_num_tst)
        print(m_XGB)
        print("Accuracy of xgboost", m_XGB.score(df_prep_x_num_tst, predict))
'''

if __name__ == "__main__":
    main()
