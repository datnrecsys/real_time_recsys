# About the project

# How to guide
1. Populate .env file based on .env.example

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