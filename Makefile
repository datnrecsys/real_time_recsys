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
	