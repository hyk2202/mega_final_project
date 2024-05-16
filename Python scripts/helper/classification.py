import inspect

from pycallgraphix.wrapper import register_method
# import logging
import numpy as np
import concurrent.futures as futures

from pandas import DataFrame, Series, concat
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier,BaggingClassifier, RandomForestClassifier
from scipy.stats import norm

from .util import my_pretty_table
from .plot import my_learing_curve, my_confusion_matrix, my_roc_curve, my_tree, my_barplot,my_plot_importance,my_xgb_tree
from .core import __ml, get_hyper_params, get_estimator

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import sys

plt.rcParams["font.family"] = (
    "AppleGothic" if sys.platform == "darwin" else "Malgun Gothic"
)
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 200
plt.rcParams["axes.unicode_minus"] = False

@register_method
def __my_classification(
    classname: any,
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    pruning: bool = False,
    **params,
) -> any:
    """분류분석을 수행하고 결과를 출력한다.

    Args:
        classname (any): 분류분석 추정기 (모델 객체)
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to True.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to True.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to True.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        any: 분류분석 모델
    """
    # ------------------------------------------------------
    # 분석모델 생성
    estimator = __ml(
        classname=classname,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        is_print=is_print,
        **params,
    )

    # ------------------------------------------------------
    # 성능평가
    my_classification_result(
        estimator,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        cv=cv,
        figsize=figsize,
        dpi=dpi,
        is_print=is_print,
    )

    # ------------------------------------------------------
    # 보고서 출력
    if report and is_print:
        my_classification_report(estimator, x_train, y_train, x_test, y_test, sort)

    return estimator

@register_method
def my_classification_result(
    estimator: any,
    x_train: DataFrame = None,
    y_train: Series = None,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve: bool = True,
    cv: int = 10,
    figsize: tuple = (12, 5),
    dpi: int = 100,
    is_print: bool = True,
) -> None:
    """회귀분석 결과를 출력한다.

    Args:
        estimator (any): 분류분석 추정기 (모델 객체)
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to True.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to True.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to True.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법(ovo, ovr, None). Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        cv (int, optional): 교차검증 횟수. Defaults to 10.
        figsize (tuple, optional): 그래프의 크기. Defaults to (12, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        is_print (bool, optional): 출력 여부. Defaults to True.
    """

    # ------------------------------------------------------
    # 성능평가 시작
    scores = []
    score_names = []

    # ------------------------------------------------------
    # 이진분류인지 다항분류인지 구분
    if hasattr(estimator, "classes_"):
        labels = list(estimator.classes_)
    elif hasattr(estimator, "n_clusters"):
        labels = list(range(estimator.n_clusters))
    elif hasattr(estimator, "n_classes_"):
        labels = list(estimator.n_classes_)
    else:
        labels = list(set(y_train))

    is_binary = len(labels) == 2

    # ------------------------------------------------------
    # 훈련데이터
    if x_train is not None and y_train is not None:
        # 추정치
        y_train_pred = estimator.predict(x_train)

        if hasattr(estimator, "predict_proba"):
            y_train_pred_proba = estimator.predict_proba(x_train)
            y_train_pred_proba_1 = y_train_pred_proba[:, 1]

        # 의사결정계수 --> 다항로지스틱에서는 사용 X
        y_train_pseudo_r2 = 0

        if is_binary and estimator.__class__.__name__ == "LogisticRegression":
            y_train_log_loss_test = -log_loss(
                y_train, y_train_pred_proba, normalize=False
            )
            y_train_null = np.ones_like(y_train) * y_train.mean()
            y_train_log_loss_null = -log_loss(y_train, y_train_null, normalize=False)
            y_train_pseudo_r2 = 1 - (y_train_log_loss_test / y_train_log_loss_null)

        # 혼동행렬
        y_train_conf_mat = confusion_matrix(y_train, y_train_pred)

        # 성능평가
        # 의사결정계수, 위양성율, 특이성, AUC는 다항로지스틱에서는 사용 불가
        # 나머지 항목들은 코드 변경 예정
        if is_binary:
            ((TN, FP), (FN, TP)) = y_train_conf_mat

            result = {
                "의사결정계수(Pseudo R2)": y_train_pseudo_r2,
                "정확도(Accuracy)": accuracy_score(y_train, y_train_pred),
                "정밀도(Precision)": precision_score(y_train, y_train_pred),
                "재현율(Recall)": recall_score(y_train, y_train_pred),
                "위양성율(Fallout)": FP / (TN + FP),
                "특이성(TNR)": 1 - (FP / (TN + FP)),
                "F1 Score": f1_score(y_train, y_train_pred),
            }

            if hasattr(estimator, "predict_proba"):
                result["AUC"] = roc_auc_score(y_train, y_train_pred_proba_1)
        else:
            result = {
                "정확도(Accuracy)": accuracy_score(y_train, y_train_pred),
                "정밀도(Precision)": precision_score(
                    y_train, y_train_pred, average="macro"
                ),
                "재현율(Recall)": recall_score(y_train, y_train_pred, average="micro"),
                "F1 Score": f1_score(y_train, y_train_pred, average="macro"),
            }

            if hasattr(estimator, "predict_proba"):
                if multiclass == "ovo" or multiclass == None:
                    result["AUC(ovo)"] = roc_auc_score(
                        y_train, y_train_pred_proba, average="macro", multi_class="ovo"
                    )

                if multiclass == "ovr" or multiclass == None:
                    result["AUC(ovr)"] = roc_auc_score(
                        y_train, y_train_pred_proba, average="macro", multi_class="ovr"
                    )

        scores.append(result)
        score_names.append("훈련데이터")

    # ------------------------------------------------------
    # 검증데이터
    if x_test is not None and y_test is not None:
        # 추정치
        y_test_pred = estimator.predict(x_test)

        if hasattr(estimator, "predict_proba"):
            y_test_pred_proba = estimator.predict_proba(x_test)
            y_test_pred_proba_1 = y_test_pred_proba[:, 1]

        # 의사결정계수
        y_test_pseudo_r2 = 0

        if is_binary and estimator.__class__.__name__ == "LogisticRegression":
            y_test_log_loss_test = -log_loss(y_test, y_test_pred_proba, normalize=False)
            y_test_null = np.ones_like(y_test) * y_test.mean()
            y_test_log_loss_null = -log_loss(y_test, y_test_null, normalize=False)
            y_test_pseudo_r2 = 1 - (y_test_log_loss_test / y_test_log_loss_null)

        # 혼동행렬
        y_test_conf_mat = confusion_matrix(y_test, y_test_pred)

        if is_binary:
            # TN,FP,FN,TP
            ((TN, FP), (FN, TP)) = y_test_conf_mat

            # 성능평가
            result = {
                "의사결정계수(Pseudo R2)": y_test_pseudo_r2,
                "정확도(Accuracy)": accuracy_score(y_test, y_test_pred),
                "정밀도(Precision)": precision_score(y_test, y_test_pred),
                "재현율(Recall)": recall_score(y_test, y_test_pred),
                "위양성율(Fallout)": FP / (TN + FP),
                "특이성(TNR)": 1 - (FP / (TN + FP)),
                "F1 Score": f1_score(y_test, y_test_pred),
            }

            if hasattr(estimator, "predict_proba"):
                result["AUC"] = roc_auc_score(y_test, y_test_pred_proba_1)
        else:
            result = {
                "정확도(Accuracy)": accuracy_score(y_test, y_test_pred),
                "정밀도(Precision)": precision_score(
                    y_test, y_test_pred, average="macro"
                ),
                "재현율(Recall)": recall_score(y_test, y_test_pred, average="macro"),
                "F1 Score": f1_score(y_test, y_test_pred, average="macro"),
            }

            if hasattr(estimator, "predict_proba"):
                if multiclass == "ovo" or multiclass == None:
                    result["AUC(ovo)"] = roc_auc_score(
                        y_test, y_test_pred_proba, average="macro", multi_class="ovo"
                    )

                if multiclass == "ovr" or multiclass == None:
                    result["AUC(ovr)"] = roc_auc_score(
                        y_test, y_test_pred_proba, average="macro", multi_class="ovr"
                    )

        scores.append(result)
        score_names.append("검증데이터")

    # ------------------------------------------------------
    # 각 항목의 설명 추가
    if is_binary:
        result = {
            "의사결정계수(Pseudo R2)": "로지스틱회귀의 성능 측정 지표로, 1에 가까울수록 좋은 모델",
            "정확도(Accuracy)": "예측 결과(TN,FP,TP,TN)가 실제 결과(TP,TN)와 일치하는 정도",
            "정밀도(Precision)": "양성으로 예측한 결과(TP,FP) 중 실제 양성(TP)인 비율",
            "재현율(Recall)": "실제 양성(TP,FN) 중 양성(TP)으로 예측한 비율",
            "위양성율(Fallout)": "실제 음성(FP,TN) 중 양성(FP)으로 잘못 예측한 비율",
            "특이성(TNR)": "실제 음성(FP,TN) 중 음성(TN)으로 정확히 예측한 비율",
            "F1 Score": "정밀도와 재현율의 조화평균",
        }

        if hasattr(estimator, "predict_proba"):
            result["AUC"] = "ROC Curve의 면적으로, 1에 가까울수록 좋은 모델"
    else:
        result = {
            "정확도(Accuracy)": "예측 결과(TN,FP,TP,TN)가 실제 결과(TP,TN)와 일치하는 정도",
            "정밀도(Precision)": "양성으로 예측한 결과(TP,FP) 중 실제 양성(TP)인 비율",
            "재현율(Recall)": "실제 양성(TP,FN) 중 양성(TP)으로 예측한 비율",
            "F1 Score": "정밀도와 재현율의 조화평균",
        }

        if hasattr(estimator, "predict_proba"):
            if multiclass == "ovo" or multiclass == None:
                result["AUC(ovo)"] = "One vs One에 대한 AUC로, 1에 가까울수록 좋은 모델"

            if multiclass == "ovr" or multiclass == None:
                result["AUC(ovr)"] = (
                    "One vs Rest에 대한 AUC로, 1에 가까울수록 좋은 모델"
                )

    scores.append(result)
    score_names.append("설명")

    # ------------------------------------------------------
    if is_print:
        print("[분류분석 성능평가]")
        result_df = DataFrame(scores, index=score_names)

        if estimator.__class__.__name__ != "LogisticRegression":
            if "의사결정계수(Pseudo R2)" in result_df.columns:
                result_df.drop(columns=["의사결정계수(Pseudo R2)"], inplace=True)

        my_pretty_table(result_df.T)

    # 결과값을 모델 객체에 포함시킴
    estimator.scores = scores[-2]

    # ------------------------------------------------------
    # 혼동행렬
    if conf_matrix and is_print:
        print("\n[혼동행렬]")

        if x_test is not None and y_test is not None:
            my_confusion_matrix(y_test, y_test_pred, figsize=figsize, dpi=dpi)
        else:
            my_confusion_matrix(y_train, y_train_pred, figsize=figsize, dpi=dpi)

    # ------------------------------------------------------
    if is_print and estimator.__class__.__name__ == "XGBClassifier":
        print("\n[변수 중요도]")
        my_plot_importance(estimator=estimator)

        feature_important = estimator.get_booster().get_score(importance_type="weight")
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = DataFrame(data=values, index=keys, columns=["score"]).sort_values(
            by="score", ascending=False
        )

        data["rate"] = data["score"] / data["score"].sum()
        data["cumsum"] = data["rate"].cumsum()

        my_pretty_table(data)

        # print("\n[TREE]")
        my_xgb_tree(booster=estimator)

    # ------------------------------------------------------
    # curve
    if is_print:
        if hasattr(estimator, "predict_proba"):
            if x_test is None or y_test is None:
                print("\n[Roc Curve]")
                my_roc_curve(
                    estimator,
                    x_train,
                    y_train,
                    hist=hist,
                    roc=roc,
                    pr=pr,
                    multiclass=multiclass,
                    dpi=dpi,
                )
            else:
                print("\n[Roc Curve]")
                my_roc_curve(
                    estimator,
                    x_test,
                    y_test,
                    hist=hist,
                    roc=roc,
                    pr=pr,
                    multiclass=multiclass,
                    dpi=dpi,
                )

        # 학습곡선
        if learning_curve:
            print("\n[학습곡선]")
            yname = y_train.name

            if x_test is not None and y_test is not None:
                y_df = concat([y_train, y_test])
                x_df = concat([x_train, x_test])
            else:
                y_df = y_train.copy()
                x_df = x_train.copy()

            x_df[yname] = y_df
            x_df.sort_index(inplace=True)

            if cv > 0:
                my_learing_curve(
                    estimator=estimator,
                    data=x_df,
                    yname=yname,
                    cv=cv,
                    figsize=figsize,
                    dpi=dpi,
                )
            else:
                my_learing_curve(
                    estimator=estimator,
                    data=x_df,
                    yname=yname,
                    figsize=figsize,
                    dpi=dpi,
                )

        if estimator.__class__.__name__ == "DecisionTreeClassifier":
            my_tree(estimator=estimator)


@register_method
def my_classification_report(
    estimator: any,
    x_train: DataFrame = None,
    y_train: Series = None,
    x_test: DataFrame = None,
    y_test: Series = None,
    sort: str = None,
) -> None:
    """분류분석 결과를 이항분류와 다항분류로 구분하여 출력한다. 훈련데이터와 검증데이터가 함께 전달 될 경우 검증 데이터를 우선한다.

    Args:
        estimator (any): 분류분석 추정기 (모델 객체)
        x_train (DataFrame, optional): 훈련 데이터의 독립변수. Defaults to None.
        y_train (Series, optional): 훈련 데이터의 종속변수. Defaults to None.
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        sort (str, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
    """
    is_binary = len(estimator.classes_) == 2

    if is_binary:
        if x_test is not None and y_test is not None:
            my_classification_binary_report(estimator, x=x_test, y=y_test, sort=sort)
        else:
            my_classification_binary_report(estimator, x=x_train, y=y_train, sort=sort)
    else:
        if x_test is not None and y_test is not None:
            my_classification_multiclass_report(
                estimator, x=x_test, y=y_test, sort=sort
            )
        else:
            my_classification_multiclass_report(
                estimator, x=x_train, y=y_train, sort=sort
            )

@register_method
def my_classification_binary_report(
    estimator: any, x: DataFrame = None, y: Series = None, sort: str = None
) -> None:
    """이항로지스틱 회귀분석 결과를 출력한다.

    Args:
        estimator (any): 분류분석 추정기 (모델 객체)
        x (DataFrame, optional): 독립변수. Defaults to None.
        y (Series, optional): 종속변수. Defaults to None.
        sort (str, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
    """
    if estimator.__class__.__name__ == "LogisticRegression":
        # 추정 확률
        y_pred_proba = estimator.predict_proba(x)

        # 추정확률의 길이(=샘플수)
        n = len(y_pred_proba)

        # 계수의 수 + 1(절편)
        m = len(estimator.coef_[0]) + 1

        # 절편과 계수를 하나의 배열로 결합
        coefs = np.concatenate([estimator.intercept_, estimator.coef_[0]])

        # 상수항 추가
        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))

        # 변수의 길이를 활용하여 모든 값이 0인 행렬 생성
        ans = np.zeros((m, m))

        # 표준오차
        for i in range(n):
            ans = (
                ans
                + np.dot(np.transpose(x_full[i, :]), x_full[i, :])
                * y_pred_proba[i, 1]
                * y_pred_proba[i, 0]
            )

        vcov = np.linalg.inv(np.matrix(ans))
        se = np.sqrt(np.diag(vcov))

        # t값
        t = coefs / se

        # p-value
        p_values = (1 - norm.cdf(abs(t))) * 2

        # VIF
        if len(x.columns) > 1:
            vif = [
                variance_inflation_factor(x, list(x.columns).index(v))
                for i, v in enumerate(x.columns)
            ]
        else:
            vif = 0

        # 결과표 생성
        xnames = estimator.feature_names_in_

        result_df = DataFrame(
            {
                "종속변수": [y.name] * len(xnames),
                "독립변수": xnames,
                "B(비표준화 계수)": np.round(estimator.coef_[0], 4),
                "표준오차": np.round(se[1:], 3),
                "t": np.round(t[1:], 4),
                "유의확률": np.round(p_values[1:], 3),
                "VIF": vif,
                "OddsRate": np.round(np.exp(estimator.coef_[0]), 4),
            }
        )

        if sort:
            if sort.upper() == "V":
                result_df = result_df.sort_values("VIF", ascending=False).reset_index()
            elif sort.upper() == "P":
                result_df = result_df.sort_values(
                    "유의확률", ascending=False
                ).reset_index()

        my_pretty_table(result_df)
    else:
        # VIF
        if len(x.columns) > 1:
            vif = [
                variance_inflation_factor(x, list(x.columns).index(v))
                for i, v in enumerate(x.columns)
            ]
        else:
            vif = 0

        # 결과표 생성
        xnames = estimator.feature_names_in_

        result_df = DataFrame(
            {
                "종속변수": [y.name] * len(xnames),
                "독립변수": xnames,
                "VIF": vif,
            }
        )

        if sort:
            if sort.upper() == "V":
                result_df = result_df.sort_values("VIF", ascending=False).reset_index()

        my_pretty_table(result_df)

@register_method
def my_classification_multiclass_report(
    estimator: any,
    x: DataFrame = None,
    y: Series = None,
    sort: str = None,
) -> None:
    """다중로지스틱 회귀분석 결과를 출력한다.

    Args:
        estimator (any): 분류분석 추정기 (모델 객체)
        x (DataFrame, optional): 독립변수. Defaults to None.
        y (Series, optional): 종속변수. Defaults to None.
        sort (str, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
    """
    class_list = list(estimator.classes_)
    class_size = len(class_list)

    if estimator.__class__.__name__ == "LogisticRegression":
        # 추정 확률
        y_pred_proba = estimator.predict_proba(x)

        # 추정확률의 길이(=샘플수)
        n = len(y_pred_proba)

        for i in range(0, class_size):
            # 계수의 수 + 1(절편)
            m = len(estimator.coef_[i]) + 1

            # 절편과 계수를 하나의 배열로 결합
            coefs = np.concatenate([[estimator.intercept_[i]], estimator.coef_[i]])

            # 상수항 추가
            x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))

            # 변수의 길이를 활용하여 모든 값이 0인 행렬 생성
            ans = np.zeros((m, m))

            # 표준오차
            for j in range(n):
                ans = (
                    ans
                    + np.dot(np.transpose(x_full[j, :]), x_full[j, :])
                    * y_pred_proba[j, i]
                )

            vcov = np.linalg.inv(np.matrix(ans))
            se = np.sqrt(np.diag(vcov))

            # t값
            t = coefs / se

            # p-value
            p_values = (1 - norm.cdf(abs(t))) * 2

            # VIF
            if len(x.columns) > 1:
                vif = [
                    variance_inflation_factor(x, list(x.columns).index(v))
                    for i, v in enumerate(x.columns)
                ]
            else:
                vif = 0

            # 결과표 생성
            xnames = estimator.feature_names_in_

            result_df = DataFrame(
                {
                    "종속변수": [y.name] * len(xnames),
                    "CLASS": [class_list[i]] * len(xnames),
                    "독립변수": xnames,
                    "B(계수)": np.round(estimator.coef_[i], 4),
                    "표준오차": np.round(se[1:], 3),
                    "t": np.round(t[1:], 4),
                    "유의확률": np.round(p_values[1:], 3),
                    "VIF": vif,
                    "OddsRate": np.round(np.exp(estimator.coef_[i]), 4),
                }
            )

            if sort:
                if sort.upper() == "V":
                    result_df.sort_values("VIF", inplace=True)
                elif sort.upper() == "P":
                    result_df.sort_values("유의확률", inplace=True)

            my_pretty_table(result_df)
    else:
        for i in range(0, class_size):
            # VIF
            if len(x.columns) > 1:
                vif = [
                    variance_inflation_factor(x, list(x.columns).index(v))
                    for i, v in enumerate(x.columns)
                ]
            else:
                vif = 0

            # 결과표 생성
            xnames = estimator.feature_names_in_

            result_df = DataFrame(
                {
                    "종속변수": [y.name] * len(xnames),
                    "CLASS": [class_list[i]] * len(xnames),
                    "독립변수": xnames,
                    "VIF": vif,
                }
            )

            if sort:
                if sort.upper() == "V":
                    result_df.sort_values("VIF", inplace=True)

            my_pretty_table(result_df)

@register_method
def my_logistic_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> LogisticRegression:
    """로지스틱 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to True.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to True.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to True.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        LogisticRegression: 회귀분석 모델
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=LogisticRegression)

    return __my_classification(
        classname=LogisticRegression,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_knn_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> KNeighborsClassifier:
    """KNN 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to True.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to True.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to True.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        KNeighborsClassifier
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=KNeighborsClassifier)

    return __my_classification(
        classname=KNeighborsClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_nb_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> GaussianNB:
    """나이브베이즈 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        GaussianNB
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=GaussianNB)

    return __my_classification(
        classname=GaussianNB,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_dtree_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    pruning: bool = False,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> DecisionTreeClassifier:
    """의사결정나무 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        DecisionTreeClassifier
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=DecisionTreeClassifier)

    return __my_classification(
        classname=DecisionTreeClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        pruning=pruning,
        **params,
    )

@register_method
def my_linear_svc_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    is_print: bool = True,
    **params,
) -> LinearSVC:
    """선형 SVM 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        LinearSVC
    """

    if "hist" in params:
        del params["hist"]
    if "roc" in params:
        del params["roc"]
    if "pr" in params:
        del params["pr"]
    if "report" in params:
        del params["report"]

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=LinearSVC)

    return __my_classification(
        classname=LinearSVC,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        is_print=is_print,
        **params,
    )

@register_method
def my_svc_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    # hist: bool = True,
    # roc: bool = True,
    # pr: bool = True,
    # multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    is_print: bool = True,
    **params,
) -> SVC:
    """SVC 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        SVC
    """

    if "hist" in params:
        del params["hist"]
    if "roc" in params:
        del params["roc"]
    if "pr" in params:
        del params["pr"]
    if "report" in params:
        del params["report"]

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=SVC)

    return __my_classification(
        classname=SVC,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        # hist=hist,
        # roc=roc,
        # pr=pr,
        # multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        is_print=is_print,
        **params,
    )

@register_method
def my_sgd_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> SGDClassifier:
    """SGD 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        SGDClassifier
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=SGDClassifier)

    return __my_classification(
        classname=SGDClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_rf_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> RandomForestClassifier:
    """랜덤포레스트 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        RandomForestClassifier
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=RandomForestClassifier)

    return __my_classification(
        classname=RandomForestClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )


@register_method
def my_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = "v",
    algorithm: list = None,
    pruning: bool = False,
    scoring: list = ["accuracy", "precision", "recall", "f1", "auc"],
    **params,
) -> DataFrame:
    """분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to False.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to False.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to False.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 독립변수 보고를 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (str, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm (list, optional): 사용하고자 하는 분류분석 알고리즘 리스트. None으로 설정할 경우 모든 알고리즘 수행 ['logistic', 'knn', 'dtree', 'svc', 'sgd']. Defaults to None.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.

    Returns:
        DataFrame: 분류분석 결과
    """

    results = []  # 결과값을 저장할 리스트
    processes = []  # 병렬처리를 위한 프로세스 리스트
    estimators = {}  # 분류분석 모델을 저장할 딕셔너리
    estimator_names = []  # 분류분석 모델의 이름을 저장할 문자열 리스트
    callstack = []

    if not algorithm:
        # algorithm = ["logistic", "knn", "dtree", "svc", "sgd", "rf"]
        algorithm = ["logistic", "knn", "dtree", "svc", "sgd"]

    if "logistic" in algorithm:
        callstack.append(my_logistic_classification)

    if "knn" in algorithm:
        callstack.append(my_knn_classification)

    if "svc" in algorithm:
        callstack.append(my_svc_classification)

    if "nb" in algorithm:
        callstack.append(my_nb_classification)

    if "dtree" in algorithm:
        callstack.append(my_dtree_classification)

    if "sgd" in algorithm:
        callstack.append(my_sgd_classification)

    if "rf" in algorithm:
        callstack.append(my_rf_classification)

    score_fields = []

    for s in scoring:
        if s == "r2":   
            score_fields.append("의사결정계수(Pseudo R2)")
        elif s == "accuracy":
            score_fields.append("정확도(Accuracy)")
        elif s == "precision":
            score_fields.append("정밀도(Precision)")
        elif s == "recall":
            score_fields.append("재현율(Recall)")
        elif s == "fallout":
            score_fields.append("위양성율(Fallout)")
        elif s == "tnr":
            score_fields.append("특이성(TNR)")
        elif s == "f1":
            score_fields.append("F1 Score")
        # elif s == "auc":
        #     score_fields.append("AUC")

    # 병렬처리를 위한 프로세스 생성 -> 분류 모델을 생성하는 함수를 각각 호출한다.
    with futures.ThreadPoolExecutor() as executor:
        for c in callstack:
            processes.append(
                executor.submit(
                    c,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    conf_matrix=conf_matrix,
                    cv=cv,
                    # hist=hist,
                    # roc=roc,
                    # pr=pr,
                    # multiclass=multiclass,
                    # learning_curve=learning_curve,
                    # report=report,
                    # figsize=figsize,
                    # dpi=dpi,
                    # sort=sort,
                    hist=False,
                    roc=False,
                    pr=False,
                    multiclass=None,
                    learning_curve=False,
                    report=False,
                    figsize=None,
                    dpi=100,
                    sort=None,
                    # pruning=pruning,
                    is_print=False,
                    # pruning=pruning,
                    **params,
                )
            )

        # 병렬처리 결과를 기다린다.
        for p in futures.as_completed(processes):
            # 각 분류 함수의 결과값(분류모형 객체)을 저장한다.
            estimator = p.result()

            if estimator is not None:
                # 분류모형 객체가 포함하고 있는 성능 평가지표(딕셔너리)를 복사한다.
                scores = estimator.scores
                # 분류모형의 이름과 객체를 저장한다.
                n = estimator.__class__.__name__
                estimator_names.append(n)
                estimators[n] = estimator
                # 성능평가 지표 딕셔너리를 리스트에 저장
                results.append(scores)

        # 결과값을 데이터프레임으로 변환
        result_df = DataFrame(results, index=estimator_names)

        if score_fields:
            result_df.sort_values(score_fields, ascending=False, inplace=True)

        my_pretty_table(result_df)

    # 최고 성능의 모델을 선택
    best_idx = result_df[score_fields[0]].idxmax()
    estimators["best"] = estimators[best_idx]

    print("\n\n==================== 최고 성능 모델: %s ====================" % best_idx)
    my_classification_result(
        estimators["best"],
        x_train,
        y_train,
        x_test,
        y_test,
        conf_matrix=conf_matrix,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        figsize=figsize,
        dpi=dpi,
        is_print=True,
    )

    my_classification_report(
        estimators["best"], x_train, y_train, x_test, y_test, sort=sort
    )

    return estimators

@register_method
def my_voting_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    hard: bool = True,
    soft: bool = True,
    lr: bool = True,
    knn: bool = True,
    nb: bool = True,
    dtree: bool = True,
    svc: bool = True,
    sgd: bool = True,
    conf_matrix: bool = True,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    is_print: bool = True,
) -> VotingClassifier:
    """Voting 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        hard (bool, optional): hard voting을 수행할지 여부. Defaults to True.
        soft (bool, optional): soft voting을 수행할지 여부. Defaults to True.
        lr (bool, optional): 로지스틱 회귀분석을 수행할지 여부. Defaults to True.
        knn (bool, optional): KNN 분류분석을 수행할지 여부. Defaults to True.
        nb (bool, optional): 나이브베이즈 분류분석을 수행할지 여부. Defaults to True.
        dtree (bool, optional): 의사결정나무 분류분석을 수행할지 여부. Defaults to True.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to False.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to False.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to False.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 독립변수 보고를 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        is_print (bool, optional): 출력 여부. Defaults to True.
    """

    params = {"voting": []}

    if hard:
        params["voting"].append("hard")

    if soft:
        params["voting"].append("soft")

    estimators = []

    if lr:
        estimators.append(("lr", get_estimator(LogisticRegression)))
        params.update(get_hyper_params(classname=LogisticRegression, key="lr"))

    if knn:
        estimators.append(("knn", get_estimator(KNeighborsClassifier)))
        params.update(get_hyper_params(classname=KNeighborsClassifier, key="knn"))

    if nb:
        estimators.append(("nb", get_estimator(GaussianNB)))
        params.update(get_hyper_params(classname=GaussianNB, key="nb"))

    if dtree:
        estimators.append(("dtree", get_estimator(DecisionTreeClassifier)))
        params.update(get_hyper_params(classname=DecisionTreeClassifier, key="dtree"))

    if svc:
        estimators.append(("svc", get_estimator(SVC)))
        params.update(get_hyper_params(classname=SVC, key="svc"))

    if sgd and soft == False:
        estimators.append(("sgd", get_estimator(SGDClassifier)))
        params.update(get_hyper_params(classname=SGDClassifier, key="sgd"))

    return __my_classification(
        classname=VotingClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        is_print=is_print,
        estimators=estimators,
        **params,
    )

@register_method
def my_bagging_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    estimator: type = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = "v",
    algorithm: list = None,
    scoring: list = ["accuracy", "precision", "recall", "f1", "auc"],
    **params,
) -> DataFrame:
    """배깅 앙상블 분류분석을 수행하고 결과를 출력한다.

    Args:
        estimator (type): 기본 분류분석 알고리즘
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to False.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to False.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to False.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 독립변수 보고를 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (str, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm (list, optional): 사용하고자 하는 분류분석 알고리즘 리스트. None으로 설정할 경우 모든 알고리즘 수행 ['logistic', 'knn', 'dtree', 'svc', 'sgd']. Defaults to None.

    Returns:
        DataFrame: 분류분석 결과
    """

    if estimator is None:
        estimator = my_classification(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            conf_matrix=False,
            cv=cv,
            hist=False,
            roc=False,
            pr=False,
            multiclass=None,
            learning_curve=False,
            report=False,
            figsize=figsize,
            dpi=dpi,
            sort=sort,
            algorithm=algorithm,
            scoring=scoring,
            **params,
        )

        estimator = estimator["best"]

    if type(estimator) is type:
        params = get_hyper_params(classname=estimator, key="estimator")
        estimator = get_estimator(classname=estimator)
    else:
        params = get_hyper_params(classname=estimator.__class__, key="estimator")

    bagging_params = get_hyper_params(classname=BaggingClassifier)
    params.update(bagging_params)

    return __my_classification(
        classname=BaggingClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        base_estimator=estimator,
        **params,
    )


@register_method
def my_ada_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    estimator: type = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = "v",
    algorithm: list = None,
    scoring: list = ["accuracy", "precision", "recall", "f1", "auc"],
    **params,
) -> AdaBoostClassifier:
    """AdaBoosting 앙상블 분류분석을 수행하고 결과를 출력한다.

    Args:
        estimator (type): 기본 분류분석 알고리즘
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to False.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to False.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to False.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 독립변수 보고를 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (str, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm (list, optional): 사용하고자 하는 분류분석 알고리즘 리스트. None으로 설정할 경우 모든 알고리즘 수행 ['logistic', 'knn', 'dtree', 'svc', 'sgd']. Defaults to None.

    Returns:
        DataFrame: 분류분석 결과
    """

    if estimator is None:
        estimator = my_classification(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            conf_matrix=False,
            cv=cv,
            hist=False,
            roc=False,
            pr=False,
            multiclass=None,
            learning_curve=False,
            report=False,
            figsize=figsize,
            dpi=dpi,
            sort=sort,
            algorithm=algorithm,
            scoring=scoring,
            **params,
        )

        estimator = estimator["best"]

    if type(estimator) is type:
        params = get_hyper_params(classname=estimator, key="estimator")
        estimator = get_estimator(classname=estimator)
    else:
        params = get_hyper_params(classname=estimator.__class__, key="estimator")

    params = get_hyper_params(classname=AdaBoostClassifier)

    return __my_classification(
        classname=AdaBoostClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        base_estimator=estimator,
        **params,
    )

@register_method
def my_gbm_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = "v",
    algorithm: list = None,
    scoring: list = ["accuracy", "precision", "recall", "f1", "auc"],
    **params,
) -> GradientBoostingClassifier:
    """GradientBoosting 앙상블 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        conf_matrix (bool, optional): 혼동행렬을 출력할지 여부. Defaults to True.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        hist (bool, optional): 히스토그램을 출력할지 여부. Defaults to False.
        roc (bool, optional): ROC Curve를 출력할지 여부. Defaults to False.
        pr (bool, optional): PR Curve를 출력할지 여부. Defaults to False.
        multiclass (str, optional): 다항분류일 경우, 다항분류 방법. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 독립변수 보고를 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (str, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm (list, optional): 사용하고자 하는 분류분석 알고리즘 리스트. None으로 설정할 경우 모든 알고리즘 수행 ['logistic', 'knn', 'dtree', 'svc', 'sgd']. Defaults to None.

    Returns:
        DataFrame: 분류분석 결과
    """
    params = get_hyper_params(classname=GradientBoostingClassifier)

    return __my_classification(
        classname=GradientBoostingClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        **params,
    )

@register_method
def my_xgb_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = "v",
    **params
) -> XGBClassifier:
    params = get_hyper_params(classname=XGBClassifier)

    return __my_classification(
        classname=XGBClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        **params,
    )


@register_method
def my_lgbm_classification(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    conf_matrix: bool = True,
    cv: int = 5,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    learning_curve=True,
    report: bool = True,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = "v",
    **params,
) -> LGBMClassifier:
    params = get_hyper_params(classname=LGBMClassifier)

    return __my_classification(
        classname=LGBMClassifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        conf_matrix=conf_matrix,
        cv=cv,
        hist=hist,
        roc=roc,
        pr=pr,
        multiclass=multiclass,
        learning_curve=learning_curve,
        report=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        **params,
    )
