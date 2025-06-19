from src.config.settings import Settings
from src.feature_pipeline.batch_processing.spark_factory import SparkLocalModeMinioSink

settings = Settings()
spark = SparkLocalModeMinioSink(settings).create_spark_session()


def main():
     # 2) Read from PostgreSQL via JDBC
    jdbc_url = f"jdbc:postgresql://{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    table_name = f"{settings.POSTGRES_OLTP_SCHEMA}.raw_metadata"

    df = (
        spark.read
             .format("jdbc")
             .option("url", jdbc_url)
             .option("dbtable", table_name)
             .option("user", settings.POSTGRES_USER)
             .option("password", settings.POSTGRES_PASSWORD)
             .option("driver", "org.postgresql.Driver")
             .load()
    )

    # 3) (Optional) Transformations
    # e.g. df = df.filter("some_column IS NOT NULL")
    # # 4) Write as a Delta table into MinIO
    output_path = f"s3a://evidently/metadata"

    (
        df.write
          .mode("overwrite")            # use "append" to incrementally add
          .save(output_path)
    )

    # spark.stop()

if __name__ == "__main__":
    main()
