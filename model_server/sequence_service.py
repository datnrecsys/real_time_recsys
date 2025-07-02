import os
import sys

import bentoml
from dotenv import load_dotenv
from models.sequence_two_tower import SequenceTwoTowerModelInput
# from model_server.models.simple_two_tower import SimpleTwoTowerModelInput

with bentoml.importing():
    import torch
    root_dir = os.path.abspath(os.path.join(__file__, "../.."))
    sys.path.insert(0, root_dir)
    # from model_server.models.sequence_two_tower import SequenceTwoTowerModelInput

load_dotenv()

model_uri = "models:/sequence_two_tower_retrieval@champion"
model_name = "sequence_two_tower_retrieval"

bentoml.mlflow.import_model(
    model_name,
    model_uri=model_uri,
    signatures={
        "predict": {"batchable": True},
    },
)

@bentoml.service(name="sequence_two_tower_retrieval_service")
class TwoTowerService:
    bento_model = bentoml.models.get(model_name)

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)
        self.inferer = self.model.unwrap_python_model()

    @bentoml.api
    def predict(self, input_data: SequenceTwoTowerModelInput):
        print(input_data.model_dump())
        rv = self.model.predict(input_data.model_dump())
        print(rv)
        return rv

    @bentoml.api
    def get_query_embeddings(self, input_data: SequenceTwoTowerModelInput):
        item_seq = [
            self.inferer.idm.get_item_index(item_id) for item_id in input_data.item_sequences[0]
        ]
        print(item_seq)
        inputs = {"item_sequence": torch.tensor([item_seq])}
        query_embedding = self.inferer.model._get_user_tower(**inputs)
        resp = {"query_embedding": query_embedding.detach().numpy().tolist()}
        return resp
