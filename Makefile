.PHONY: test test-scientific smoke-test run-benchmark build-frontend docker-build lint

test:
	python -m pytest tests/ -v

test-scientific:
	python -m pytest tests/scientific/ -v

smoke-test:
	python scripts/smoke_test.py

run-benchmark:
	python -m src.experiments.benchmark --config configs/smoke.yaml

build-frontend:
	cd frontend && npm ci && npm run lint && npm run build

docker-build:
	docker build -t pinnflow-api -f Dockerfile.api .
	docker build -t pinnflow-ui -f Dockerfile.ui .
