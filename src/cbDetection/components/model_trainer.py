import os
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from cbDetection.entity import ModelTrainerConfig
from cbDetection import logger
from cbDetection.utils.exception import CustomException
from cbDetection.utils.ml_helper import save_object, evaluate_models


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def initiate_model_trainer(self, train_data: tuple, validation_data: tuple):
        logger.info("Starting model evaluation")
        try:
            # Unbox the training and testing data
            X_train, y_train = train_data[0], train_data[1]
            X_test, y_test = validation_data[0], validation_data[1] 
            
            # Define models and hyperparameters 
            models_dict = {
                "XGBClassifier": XGBClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42),
                "Naive Bayes": MultinomialNB(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
            }
            params_dict = {
                "XGBClassifier": self.config.params_xgboost,
                "Logistic Regression": self.config.params_logistic_regression,
                "Naive Bayes": self.config.params_naive_bayes,
                "Decision Tree": self.config.params_decision_tree,
                "Random Forest": self.config.params_random_forest,
            }

            logger.info(f"Starting model performance assessment...")
            # Evaluate all models
            model_report = evaluate_models(models_dict=models_dict, params_dict=params_dict,
                                           X=X_train, y=y_train,
                                           validation_data=(X_test, y_test))
            
            for model, score in model_report.items():
                print("[{}] - Accuracy: {:.4f}".format(model, score))
            
            ## To get best model name from dict
            best_model_name, best_model_score = max(model_report.items(), key=lambda x: x[1])
            logger.info(f"Best model: {best_model_name} with score: {best_model_score:.4f}")

            # the best model
            best_model = models_dict[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("There is no model that achieves satisfactory performance.")
            
            # Save the best model
            save_object(
                file_path=os.path.join(self.config.root_dir, self.config.model_name),
                obj=best_model
            )
            logger.info(f"Best model saved to {self.config.model_name}")
            
            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            return acc_score
        except Exception as e:
            raise CustomException(e, sys)