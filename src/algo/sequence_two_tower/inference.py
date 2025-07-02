import numpy as np
import torch 

import mlflow
from src.utils.embedding_id_mapper import IDMapper
from src.domain.model_request import SequenceModelRequest
class SequenceRatingPredictionInferenceWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device  = next(model.parameters()).device
        
    def load_context(self, context):
        
        json_path = context.artifacts.get("idm_path")
        self.device = context.model_config.get("device", self.device)
        self.model.to(self.device)
        self.idm = IDMapper().load(json_path)
        
    def predict(self, context, model_input, params=None):
        import numpy as np
        """
        Predict ratings for a sequence of items.
        
        Args:
            context: The context containing artifacts and parameters.
            model_input: A DataFrame with columns 'user_id' and 'item_sequence'.
            params: Additional parameters (not used here).
        
        Returns:
            A DataFrame with predicted ratings.
        """
        # Convert model to device
        sequence_length = 10
        padding_value = -1
        
        if not isinstance(model_input, dict):
            # This is to work around the issue where MLflow automatically convert dict input into Dataframe
            # Ref: https://github.com/mlflow/mlflow/issues/11930
            model_input = model_input.to_dict(orient="records")[0]

        # user_indices = [self.idm.get_user_index(id_) for id_ in model_input['user_ids']]
        item_indices = [self.idm.get_item_index(id_) for id_ in model_input["item_ids"]]
        item_sequences = []
        
        for item_sequence in model_input["item_sequences"]:
            item_sequence = [self.idm.get_item_index(item) for item in item_sequence]
            padding_needed = sequence_length - len(item_sequence)
            item_sequence = np.pad(
                item_sequence,
                (padding_needed, 0),
                mode='constant',
                constant_values=padding_value
            )
            item_sequences.append(item_sequence)

        infer_output = self.infer(item_sequences, item_indices)
        infer_output = infer_output.tolist()
        
        return {
            **model_input,
            "scores": infer_output,
        }
        
    def infer(self, item_sequences, item_indices):
        item_sequences = torch.tensor(item_sequences, device=self.device)
        item_indices = torch.tensor(item_indices, device=self.device)

        input_data = SequenceModelRequest(
            item_sequence=item_sequences,
            target_item=item_indices,
            recommendation=False,
            
        )

        output = self.model.predict(input_data)
        return output.view(len(item_indices)).detach().cpu().numpy()