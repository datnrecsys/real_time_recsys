.PHONY:
.ONESHELL:

include .env
export

infra-up:
	docker compose -f ./compose.yml up -d

infra-logs:
	docker compose -f ./compose.yml logs

infra-down:
	docker compose -f ./compose.yml down -v
	
clean-raw-data:
	rm -rf ./data_for_ai/raw/*.parquet

build-pipeline:
	docker build -f feature_pipeline.Dockerfile . -t recsys-mvp-pipeline:0.0.1

feature-server-up:
	docker compose -f compose.yml up -d feature_online_server feature_offline_server feature_store_ui --force-recreate

feature-server-down:
	docker compose -f compose.yml down -v

airflow-up:
	docker compose -f compose.airflow.yml up -d

airflow-logs:
	docker compose -f compose.airflow.yml logs
airflow-down:
	docker compose -f compose.airflow.yml down -v