import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
mlflow.set_experiment('model-lifecycle')
with mlflow.start_run() as run:
    n_estimators=200
    model=RandomForestClassifier(n_estimators=n_estimators,random_state=42)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte); proba=model.predict_proba(Xte)[:,1]
    acc=accuracy_score(yte,pred); auc=roc_auc_score(yte,proba)
    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_metrics({'accuracy':acc,'roc_auc':auc})
    mlflow.sklearn.log_model(model,'model')
    print('run_id:',run.info.run_id,'accuracy:',round(acc,4),'roc_auc:',round(auc,4))
