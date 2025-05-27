import mlflow
from src.utils.embedding_id_mapper import IDMapper
import torch
import numpy 

class TwoTowerInferenceWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for the TwoTowerRating model to enable MLflow model serving.
    """

    def __init__(self, model):
        """
        Initialize the wrapper with the model and ID mappers.

        Args:
            model (TwoTowerRating): The TwoTowerRating model.
        """
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        
    def load_context(self, context):
        """
        Load the context for the model serving.

        Args:
            context (mlflow.pyfunc.PythonFunctionContext): The context for the model serving.
        """
        # Load ID mappers from the context
        idm_json_path = context.artifacts["idm"]
        self.idm = IDMapper().load(idm_json_path)
        
    def predict(self, context, model_input, params = None):
        """
        Predict ratings for the given user-item pairs.
        """
        
        if not isinstance(model_input, dict):
            model_input = model_input.to_dict(orient="records")[0]

        # Convert user and item IDs to indices
        user_indices = [
            self.idm.get_user_index(_user_id) for _user_id in model_input["user_id"]
        ]
        item_indices = [
            self.idm.get_item_index(_item_id) for _item_id in model_input["item_id"]
        ]

        predictions = self.infer(user_indices, item_indices).tolist()
        
        return {
            "user_id": model_input["user_id"],
            "item_id": model_input["item_id"],
            "scores": predictions,
        }

        
    def infer(self, user_indices, item_indices):
        """
        Perform inference using the model.

        Args:
            user_indices (list): List of user indices.
            item_indices (list): List of item indices.

        Returns:
            torch.Tensor: Predicted interaction scores.
        """
        # Convert to tensors
        user_tensor = torch.tensor(user_indices, dtype=torch.long)
        item_tensor = torch.tensor(item_indices, dtype=torch.long)

        
        predictions = self.model.predict(user_tensor, item_tensor)

        return predictions.detach().numpy()