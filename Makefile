.PHONY:
.ONESHELL:

include .env
export

infra-up:
	docker compose -f ./compose.yml up -d

infra-logs:
	docker compose -f ./compose.yml logs

down:
	docker compose -f ./compose-postgres.yaml down && docker volume remove hm-scalablerecs_recsys-dwh