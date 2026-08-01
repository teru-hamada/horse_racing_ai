"""用途別パッケージを通常のimport文から利用する窓口。"""

from importlib import import_module


_storage = import_module("src.00_common.storage")
_html_jobs = import_module(
    "src.10_scrapers_html_collection.html_collection_jobs"
)
_html_scraper = import_module(
    "src.10_scrapers_html_collection.scrapers_html_collection_netkeiba"
)
_database_jobs = import_module(
    "src.20_scrapers_database_creation.database_creation_jobs"
)
_database_scraper = import_module(
    "src.20_scrapers_database_creation.scrapers_database_creation_netkeiba"
)
_demo_data = import_module(
    "src.20_scrapers_database_creation.demo_data"
)
_logging = import_module("src.00_common.logging_utils")
_config = import_module("src.00_common.config")
_features = import_module("src.30_ai_modeling.common.features")
_top3_target = import_module("src.30_ai_modeling.tasks.top3.target")
_modeling = import_module("src.30_ai_modeling.service")
_model_registry = import_module("src.30_ai_modeling.registry")

cancel_html_collection_job = _html_jobs.cancel_job
list_html_collection_jobs = _html_jobs.list_jobs
start_html_collection_job = _html_jobs.start_job

cancel_database_job = _database_jobs.cancel_database_job
list_database_jobs = _database_jobs.list_database_jobs
start_database_job = _database_jobs.start_database_job

NetkeibaHtmlCollector = _html_scraper.NetkeibaHtmlCollector
NetkeibaDatabaseCreator = _database_scraper.NetkeibaDatabaseCreator
generate_demo_records = _demo_data.generate_demo_records
AppLogger = _logging.AppLogger
PATHS = _config.PATHS
AppPaths = _config.AppPaths
get_paths = _config.get_paths

RACE_RECORD_COLUMNS = _storage.RACE_RECORD_COLUMNS
connect = _storage.connect
empty_race_frame = _storage.empty_race_frame
normalize_race_frame = _storage.normalize_race_frame
save_collection_run = _storage.save_collection_run
save_race_records = _storage.save_race_records
load_records = _storage.load_records
race_record_summary = _storage.race_record_summary
race_record_years = _storage.race_record_years
collection_runs = _storage.collection_runs
save_model_run = _storage.save_model_run
model_runs = _storage.model_runs
save_prediction_run = _storage.save_prediction_run
dashboard_summary = _storage.dashboard_summary
json_dump = _storage.json_dump

NUMERIC_FEATURES = _features.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _features.CATEGORICAL_FEATURES
MODEL_FEATURES = _features.MODEL_FEATURES
add_historical_features = _features.add_historical_features
build_training_frame = _top3_target.build_training_frame
build_prediction_frame = _features.build_prediction_frame

TrainConfig = _modeling.TrainConfig
train_model = _modeling.train_model
predict_race = _modeling.predict_race
predict_historical_race = _modeling.predict_historical_race
train_task = _modeling.train_task
predict_task = _modeling.predict_task
predict_historical_task = _modeling.predict_historical_task
MODEL_TASKS = _model_registry.MODEL_TASKS
get_model_task = _model_registry.get_task
