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
_feature_engineering = import_module(
    "src.30_ai_modeling.feature_engineering"
)
_feature_storage = import_module(
    "src.30_ai_modeling.feature_engineering.storage"
)
_feature_jobs = import_module(
    "src.30_ai_modeling.feature_engineering.jobs"
)
_speed_jobs = import_module(
    "src.30_ai_modeling.feature_engineering.speed_jobs"
)
_recent_speed_jobs = import_module(
    "src.30_ai_modeling.feature_engineering.recent_speed_jobs"
)
_race_entry_jobs = import_module(
    "src.30_ai_modeling.feature_engineering.race_entry_jobs"
)
_training_dataset = import_module("src.30_ai_modeling.training_dataset")
_static_site = import_module("src.static_site")
_prediction_comparison = import_module("src.prediction_comparison")
_weather_forecast = import_module("src.weather_forecast")

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
predict_race_date = _modeling.predict_race_date
train_task = _modeling.train_task
predict_task = _modeling.predict_task
predict_historical_task = _modeling.predict_historical_task
predict_date_task = _modeling.predict_date_task
MODEL_TASKS = _model_registry.MODEL_TASKS
get_model_task = _model_registry.get_task

FeatureContext = _feature_engineering.FeatureContext
FeatureGenerator = _feature_engineering.FeatureGenerator
FeaturePipeline = _feature_engineering.FeaturePipeline
FeatureRegistry = _feature_engineering.FeatureRegistry
FeatureSetDefinition = _feature_engineering.FeatureSetDefinition
RecentFormGenerator = _feature_engineering.RecentFormGenerator
RecentSpeedGenerator = _feature_engineering.RecentSpeedGenerator
RecentSpeedRunConfig = _feature_engineering.RecentSpeedRunConfig
generate_recent_speed_features = _feature_engineering.generate_recent_speed_features
RaceEntryGenerator = _feature_engineering.RaceEntryGenerator
RaceEntryRunConfig = _feature_engineering.RaceEntryRunConfig
generate_race_entry_features = _feature_engineering.generate_race_entry_features
SpeedIndexGenerator = _feature_engineering.SpeedIndexGenerator
SpeedIndexRunConfig = _feature_engineering.SpeedIndexRunConfig
generate_speed_index_features = _feature_engineering.generate_speed_index_features
RecentFormRunConfig = _feature_engineering.RecentFormRunConfig
generate_recent_form_features = (
    _feature_engineering.generate_recent_form_features
)
prepare_historical_performances = (
    _feature_engineering.prepare_historical_performances
)
prepare_speed_performances = _feature_engineering.prepare_speed_performances
calculate_elapsed_days = _feature_engineering.calculate_elapsed_days
decay_weight = _feature_engineering.decay_weight
decay_weights = _feature_engineering.decay_weights
effective_count = _feature_engineering.effective_count
weighted_mean = _feature_engineering.weighted_mean
weighted_sum = _feature_engineering.weighted_sum
connect_feature_store = _feature_storage.connect
save_feature_run = _feature_storage.save_feature_run
save_features = _feature_storage.save_features
replace_features = _feature_storage.replace_features
replace_performance_features = _feature_storage.replace_performance_features
load_performance_features = _feature_storage.load_performance_features
clear_performance_features = _feature_storage.clear_performance_features
performance_feature_summary = _feature_storage.performance_feature_summary
load_features = _feature_storage.load_features
clear_features = _feature_storage.clear_features
feature_runs = _feature_storage.feature_runs
feature_store_summary = _feature_storage.feature_store_summary
feature_freshness = _feature_engineering.feature_freshness
performance_feature_freshness = (
    _feature_engineering.performance_feature_freshness
)
recent_speed_freshness = _feature_engineering.recent_speed_freshness
source_data_state = _feature_engineering.source_data_state
source_state_token = _feature_engineering.source_state_token
start_feature_generation_job = _feature_jobs.start_feature_generation_job
cancel_feature_generation_job = _feature_jobs.cancel_feature_generation_job
list_feature_generation_jobs = _feature_jobs.list_feature_generation_jobs
start_speed_index_job = _speed_jobs.start_speed_index_job
cancel_speed_index_job = _speed_jobs.cancel_speed_index_job
list_speed_index_jobs = _speed_jobs.list_speed_index_jobs
start_recent_speed_job = _recent_speed_jobs.start_recent_speed_job
cancel_recent_speed_job = _recent_speed_jobs.cancel_recent_speed_job
list_recent_speed_jobs = _recent_speed_jobs.list_recent_speed_jobs
start_race_entry_job = _race_entry_jobs.start_race_entry_job
cancel_race_entry_job = _race_entry_jobs.cancel_race_entry_job
list_race_entry_jobs = _race_entry_jobs.list_race_entry_jobs
TrainingFeatureSet = _training_dataset.TrainingFeatureSet
TrainingDatasetConfig = _training_dataset.TrainingDatasetConfig
FeatureSetBuildReport = _training_dataset.FeatureSetBuildReport
TrainingDataset = _training_dataset.TrainingDataset
TrainingDatasetBuilder = _training_dataset.TrainingDatasetBuilder
build_prediction_site = _static_site.build_prediction_site
initialize_prediction_site = _static_site.initialize_prediction_site
prediction_date_status = _static_site.prediction_date_status
latest_prediction_file = _static_site.latest_prediction_file
compare_prediction_with_finish = (
    _prediction_comparison.compare_prediction_with_finish
)
compare_prediction_date = _prediction_comparison.compare_prediction_date
fetch_jravan_weather = _weather_forecast.fetch_jravan_weather
parse_jravan_forecast = _weather_forecast.parse_jravan_forecast
weather_to_track_condition = _weather_forecast.weather_to_track_condition
