import matplotlib.pyplot as plt

from dispatcher import WSDispatcher
from pandas import read_csv as pd_read_csv
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import train_test_split
from numpy import average as np_average
from numpy import array as np_array

in_source = 'kuleuven'
dispatcher = WSDispatcher(source=in_source)
se_df = dispatcher.academic_clusterer.semesters_features
sf_df = dispatcher.academic_clusterer.students_features
ss_df = pd_merge( se_df, sf_df, on='student' )

FEATURES = ['factor1_measure', 'factor2_measure', 'factor3_measure',
            'factor4_measure', 'factor5_measure', 'factor6_measure',
            'alpha_total', 'beta_total', 'skewness_total', 'year_n',
            'courses_num']


X = ss_df[FEATURES+['_class']].as_matrix()
y = ss_df['ha_reprobado'].apply(lambda x: 0 if x else 1).values
#y = ss_df['_class'].values

# split train, test for calibration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

_classes = range(4)

weights = []
for _class in _classes:
    tmp_mask = X_train[:, 10]==_class
    sample = X_train[ tmp_mask ]
    sample_y = y_train[ tmp_mask ]
    weights.append( len( sample_y[ sample_y == 0 ] )*1.0 / len( sample ) )

def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
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
    for clf, name in [(lr, 'Logistic'),
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
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

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

# Plot calibration cuve for Gaussian Naive Bayes
plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

# Plot calibration cuve for Linear SVC
plot_calibration_curve(LinearSVC(), "SVC", 2)

plt.show()

################################################################################

abs_df = pd_read_csv('../../data/kuleuven/abs_df_8763336794416746037.csv',
                     index_col=0)
abs_df = abs_df.fillna(-1000)
ha_df = pd_read_csv('../../data/kuleuven/students_courses.csv',
                    index_col=0)
sha_df = pd_merge( ha_df, sf_df, on='student' )
sha_df = pd_merge( sha_df, abs_df, on='course' )

FEATURES = ['factor1_measure', 'factor2_measure', 'factor3_measure',
            'factor4_measure', 'factor5_measure', 'factor6_measure',
            'alpha', 'beta', 'skewness'
            ]

X = sha_df[FEATURES].as_matrix()
y = sha_df['status'].apply(lambda x: 0 if x=='Failed' else 1).values

# split train, test for calibration
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=7)

# Plot calibration cuve for Gaussian Naive Bayes
#plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

# Plot calibration cuve for Linear SVC
plot_calibration_curve(LinearSVC(), "SVC", 2)

plt.show()

def probability_of_union(probs):
    x = 0.0
    for prob in probs:
        x += prob*( 1 - x )
    return x

def plot_calibration_curve_with_union_prob(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    # isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    # sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')
    # clf = LinearSVC()
    # sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
    # lr = sigmoid

    lr.fit(X, y)

    
    data_X = ss_df.index
    data_y = ss_df['ha_reprobado'].apply(lambda x: 0 if x else 1).values

    # split train, test for calibration
    X_tra_i, X_te_i, y_tra_i, y_te_i = train_test_split(data_X,
                                                        data_y,
                                                        test_size=0.30,
                                                        random_state=7)

    lr_predict_proba = lr.predict_proba
    lr_predict = lr.predict

    def ss_predict(row):
        features = ['factor1_measure', 'factor2_measure', 'factor3_measure',
                    'factor4_measure', 'factor5_measure', 'factor6_measure'
                    ]
        data = row[ features ].values.tolist()

        probs = []
        preds = False
        featur_s = ['alpha','beta','skewness']
        probs_append = probs.append
        for course in row['taken_courses'].split(' '):
            data_t = list( data )
            tmp = abs_df[ abs_df['course']==course ][ featur_s ].values.tolist()[0]
            data_t.extend( tmp )
            prob = lr_predict_proba( [data_t] )[:, 1]
            pred = lr_predict( [data_t] )[0]
            preds = True if pred==0 else False
            probs_append( prob[0] )
        return [ probability_of_union( probs ), preds ]

    pred_ = np_array( ss_df.ix[ X_te_i ].apply( ss_predict, axis=1 ).values.tolist() )
    y_pred = pred_[:, 1]
    prob_pos = pred_[:, 0]
        
        
    

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    clf = lr; name = 'Logistic Regression'
    
    clf_score = brier_score_loss(y_te_i, prob_pos, pos_label=data_y.max())
    print("%s:" % name)
    print("\tBrier: %1.3f" % (clf_score))
    print("\tPrecision: %1.3f" % precision_score(y_te_i, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_te_i, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_te_i, y_pred))

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_te_i, prob_pos, n_bins=10)

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

# Plot calibration cuve for Linear SVC
plot_calibration_curve_with_union_prob(LinearSVC(), "SVC", 2)

plt.show()
