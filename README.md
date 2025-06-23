# About the project

# How to guide
1. Populate .env file based on .env.example
    ```sh
    set -a && source .env && set +a
    ```

2. Create .venv and install packages
    ```sh
    conda create --prefix .venv python=3.11.9
    poetry install
    ```

3. Build up essential infra
    ```sh
    make infra-up
    ```

    Make sure to check logs to verify everything works

    ```sh
    make infra-logs
    ```

4. Configure dvc
    ```sh
    cat <<EOF > .dvc/config
    [core]
        remote = minio
    ['remote "minio"']
        url = s3://data/dvc_remote
        endpointurl = http://$MINIO_HOST:$MINIO_PORT
        access_key_id = $AWS_ACCESS_KEY_ID
        secret_access_key = $AWS_SECRET_ACCESS_KEY
    EOF
    ```

5. Pull data
    ```sh
    poetry run dvc pull
    ```

6. DBT
    ```sh
    cd real_time_recsys/feature_pipeline/dbt/feature_store
    cat <<EOF > profiles.yml
    feature_store:
    outputs:
        dev:
        dbname: $POSTGRES_DB
        host: $POSTGRES_HOST
        pass: $POSTGRES_PASSWORD
        port: $POSTGRES_PORT
        schema: $POSTGRES_FEATURE_STORE_OFFLINE_SCHEMA
        threads: 1
        type: postgres
        user: $POSTGRES_USER
    target: dev
    EOF'
    ```


export MATERIALIZE_CHECKPOINT_TIME=$(poetry run python scripts/check_oltp_max_timestamp.py 2>&1 | awk -F'<ts>|</ts>' '{print $2}')