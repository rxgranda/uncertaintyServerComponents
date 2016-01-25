################################################################################
import matplotlib.pyplot as plt

from dispatcher import WSDispatcher
from pandas import read_csv as pd_read_csv
from pandas import merge as pd_merge

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import train_test_split
from numpy import average as np_average
from numpy import array as np_array
from skfuzzy import cmeans, cmeans_predict
from data_loader import kuleuven_loader
from itertools import combinations

in_source = 'kuleuven'
dispatcher = WSDispatcher(source=in_source)
se_df = dispatcher.academic_clusterer.semesters_features
sf_df = dispatcher.academic_clusterer.students_features
ss_df = pd_merge( se_df, sf_df, on='student' )

# cd doc/calibration_test/

abs_df = pd_read_csv('../../data/kuleuven/abs_df_1716653621.csv', index_col=0)
abs_df = abs_df.fillna(-1000)
ha_df = pd_read_csv('../../data/kuleuven/students_courses.csv', index_col=0)
ha_df = ha_df.drop_duplicates(['year','status','course','grade','student'])
sha_df = pd_merge( ha_df, sf_df, on='student' )
sha_df = pd_merge( sha_df, abs_df, on='course' )

OP = []
OP_append = OP.append


def plot_calibration_curve_from_data(X, y, est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.30,
                                                        random_state=7)
    
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic Regression'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')
                      ]:
        clf.fit(X_train, y_train)
        # clf.fit(X_train[:,:10], X_train[:, 10])
        y_pred = clf.predict(X_test)
        # y_pred = clf.predict(X_test[:,:10])
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
            # prob_pos = clf.predict_proba(X_test[:,:10])[:, 1]
            # prob_pos = clf.predict_proba(X_test[:,:10])[:, 1]*weights[1]
            # prob_pos = np_average( 1 - clf.predict_proba(X_test[:,:10]), axis=1, weights=weights )
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            # prob_pos = clf.decision_function(X_test[:,:10])[:, 1]
            # prob_pos = clf.decision_function(X_test[:,:10])[:, 1]*weights[1]
            # prob_pos = np_average( 1 - clf.decision_function(X_test[:,:10]), axis=1, weights=weights )
        prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("* %s:" % name)
        OP_append("* %s:" % name)
        print(" * Brier: %1.3f" % (clf_score))
        OP_append(" * Brier: %1.3f" % (clf_score))
        print(" * Precision: %1.3f" % precision_score(y_test, y_pred))
        OP_append(" * Precision: %1.3f" % precision_score(y_test, y_pred))
        print(" * Recall: %1.3f" % recall_score(y_test, y_pred))
        OP_append(" * Recall: %1.3f" % recall_score(y_test, y_pred))
        print(" * F1: %1.3f\n" % f1_score(y_test, y_pred))
        OP_append(" * F1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    fig.savefig('NF/%s.png'%name, dpi=fig.dpi)


gpa_df = pd_read_csv('../../data/kuleuven/gpa_df.csv', index_col=0)
ss_df = pd_merge( ss_df, gpa_df, on='student' )
cs_df = kuleuven_loader.cs_df
cs_df['course'] = cs_df['code']
abs_df = dispatcher.academic_clusterer.courses_features


FEATURES = ['GPA','performance','credits_total','alpha_total','beta_total']
CLASSIFIERS = [#( SVC(), 'RBF SVM'),
               ( LinearSVC(), 'Linear SVM'),
               ( GaussianNB(), 'Naive Bayes'),
               ( KNeighborsClassifier(), 'Nearest Neighbors'),
               #( RandomForestClassifier(), 'Random Forest'),
               #( AdaBoostClassifier(), 'AdaBoost'),
               #( DecisionTreeClassifier(), 'Decision Tree')
               ]

def get_credits_total(semester):
    credit_total = abs_df[ abs_df['course'].isin(semester) ]['credits'].sum()
    return credit_total

ss_df['credits_total'] = ss_df.apply( get_credits_total, axis=1 )

def a_combinations(a):
    _r = []
    for i in xrange(2,len(a)+1):
        _r.extend(list( combinations(a,i) ) )
    return _r

y = ss_df['ha_reprobado'].apply(lambda x: 0 if x else 1).values

for features in a_combinations(FEATURES):
    data = ss_df[ list(features) ].as_matrix()
    X = data

    for clf, name in CLASSIFIERS:
        plot_calibration_curve_from_data(X,
                                         y,
                                         clf,
                                         '%s_%s'%(name, features), 1)
