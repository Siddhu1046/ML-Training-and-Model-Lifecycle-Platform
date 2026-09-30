import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT','model-lifecycle'))

def train():
    X,y=load_breast_cancer(return_X_y=True)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    with mlflow.start_run() as run:
        n_estimators=200
        model=RandomForestClassifier(n_estimators=n_estimators,random_state=42)
        model.fit(Xtr,ytr)
        pred=model.predict(Xte); proba=model.predict_proba(Xte)[:,1]
        metrics={'accuracy':accuracy_score(yte,pred),'f1':f1_score(yte,pred),'roc_auc':roc_auc_score(yte,proba)}
        mlflow.log_param('n_estimators',n_estimators)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model,'model',registered_model_name='LifecycleClassifier')
        os.makedirs('artifacts',exist_ok=True); joblib.dump(model,'artifacts/model.joblib')
        print('run_id:',run.info.run_id, metrics)

if __name__=='__main__': train()
