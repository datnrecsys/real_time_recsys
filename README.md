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
    dvc pull
    ```