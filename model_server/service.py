import os
import sys

import bentoml
from dotenv import load_dotenv

# from model_server.models.simple_two_tower import SimpleTwoTowerModelInput

with bentoml.importing():
    root_dir = os.path.abspath(os.path.join(__file__, "../.."))
    sys.path.insert(0, root_dir)
    from model_server.models.simple_two_tower import SimpleTwoTowerModelInput

load_dotenv()

model_uri = "models:/two-tower@champion"
model_name = "two-tower"

bentoml.mlflow.import_model(
    model_name,
    model_uri=model_uri,
    signatures={
        "predict": {"batchable": True},
    },
)


@bentoml.service(name="two_tower_service")
class TwoTowerService:
    bento_model = bentoml.models.get(model_name)

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data: SimpleTwoTowerModelInput):
        print(input_data.model_dump())
        rv = self.model.predict(input_data.model_dump())
        print(rv)
        return rv
