from pyspark.sql import SparkSession
from abc import ABC, abstractmethod
from loguru import logger
from src.config.settings import Settings

class SparkFactory(ABC):

    @abstractmethod
    def create_spark_session(self, app_name: str ) -> SparkSession:
        """
        Create a Spark session with the specified application name.
        
        Args:
            app_name (str): The name of the Spark application.
        
        Returns:
            SparkSession: A Spark session instance.
        """
        ...
    
class SparkLocalModeMinioSink(SparkFactory):
    
    def __init__(self, settings: Settings= Settings()):
        """
        Initialize the SparkLocalModeMinioSink with the provided settings.
        
        Args:
            settings (Settings): Configuration settings for Spark and MinIO.
        """
        self.settings = settings
        
    def create_spark_session(self, app_name: str = "MinioSinkSparkApp") -> SparkSession:
        import os
        # Clear Hadoop configuration directory to avoid default configs
        os.environ.pop("HADOOP_CONF_DIR", None)
        os.environ.pop("HADOOP_HOME", None)
        spark = (
        SparkSession.builder
            .appName(app_name)
            # PostgreSQL driver
            .config(
           "spark.jars.packages",
          ",".join([
            "org.postgresql:postgresql:42.5.1",
            "org.apache.hadoop:hadoop-aws:3.3.2",
             "com.amazonaws:aws-java-sdk-bundle:1.11.1026"
          ])
       )
            # S3A / MinIO settings
            # S3A / MinIO settings
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.endpoint", f"{self.settings.MINIO_ENDPOINT}")
            .config("spark.hadoop.fs.s3a.access.key", self.settings.MINIO_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", self.settings.MINIO_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            # Override all timeout configurations with numeric values (milliseconds)
            .config("spark.hadoop.fs.s3a.connection.timeout", "200000")
            .config("spark.hadoop.fs.s3a.socket.timeout", "200000")
            .config("spark.hadoop.fs.s3a.request.timeout", "200000")
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "200000")
            .config("spark.hadoop.fs.s3a.retry.limit", "3")
            .config("spark.hadoop.fs.s3a.retry.interval", "1000")
            .config("spark.hadoop.fs.s3a.connection.idle.timeout", "60000")
            .config("spark.hadoop.fs.s3a.timeout.read", "200000")
            .config("spark.hadoop.fs.s3a.timeout.write", "200000")
            # Override configurations that might contain "24h" or other time units
            .config("spark.hadoop.fs.s3a.s3guard.cli.prune.age", "86400000")  # 24h in milliseconds
            .config("spark.hadoop.fs.s3a.metadatastore.metadata.ttl", "86400000")  # 24h in milliseconds
            .config("spark.hadoop.fs.s3a.directory.marker.retention", "keep")
            .config("spark.hadoop.fs.s3a.bucket.probe", "0")
            # Disable SSL for MinIO
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            # Use simple credentials provider
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            # Disable fast upload which can cause issues
            .config("spark.hadoop.fs.s3a.fast.upload", "false")
            .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
            .config("spark.hadoop.fs.s3a.multipart.threshold", "134217728")
            # Override any default configurations that might contain time units
            .config("spark.hadoop.fs.s3a.connection.maximum", "15")
            .config("spark.hadoop.fs.s3a.threads.max", "10")
            .config("spark.hadoop.fs.s3a.threads.core", "5")
            .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000")  # in milliseconds
            # Disable S3Guard and other features that might have time configurations
            .config("spark.hadoop.fs.s3a.metadatastore.impl", "org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore")
            .config("spark.hadoop.fs.s3a.s3guard.ddb.table.create", "false")
            .getOrCreate()
        )
        return spark
