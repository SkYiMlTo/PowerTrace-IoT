-- PowerDetect Dataset - TimescaleDB schema
-- LA-DT ingestion backend (strict typing + temporal integrity)

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- -------------------------------------------------------------------
-- Device metadata dimension table
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS devices (
    node_id SMALLINT PRIMARY KEY,
    device_name TEXT NOT NULL UNIQUE,
    device_type TEXT NOT NULL DEFAULT 'sensor',
    location TEXT,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO devices (node_id, device_name, location, description)
VALUES
    (1, 'pylon_sensor_01', 'Pylon A1 - Ligne 315kV', 'ESP32 + DHT11 + MPU6500 + ACS712-5A'),
    (2, 'pylon_sensor_02', 'Pylon B1 - Ligne 315kV', 'ESP32 + DHT11 + MPU6500 + ACS712-5A'),
    (3, 'pylon_sensor_03', 'Pylon A2 - Ligne 315kV', 'ESP32 + DHT11 + MPU6500 + ACS712-5A'),
    (4, 'pylon_sensor_04', 'Pylon B2 - Ligne 315kV', 'ESP32 + DHT11 + MPU6500 + ACS712-5A'),
    (5, 'pylon_sensor_05', 'Pylon A3 - Ligne 315kV', 'ESP32 + DHT11 + MPU6500 + ACS712-5A')
ON CONFLICT (node_id) DO UPDATE SET
    device_name = EXCLUDED.device_name,
    location = EXCLUDED.location,
    description = EXCLUDED.description;

-- -------------------------------------------------------------------
-- Power telemetry hypertable (primary ML training signal)
-- Ground-truth label attack_state is indexed for fast filtering/windowing.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS power_readings (
    event_time TIMESTAMPTZ NOT NULL,
    topic TEXT NOT NULL,
    node_id SMALLINT NOT NULL REFERENCES devices(node_id),
    device_name TEXT NOT NULL,
    attack_state TEXT NOT NULL DEFAULT 'none',
    sensor_voltage_mean_v DOUBLE PRECISION NOT NULL,
    sensor_voltage_peak_v DOUBLE PRECISION NOT NULL,
    absolute_current_ma DOUBLE PRECISION NOT NULL,
    variance_current_ma DOUBLE PRECISION NOT NULL,
    peak_current_ma DOUBLE PRECISION NOT NULL,
    current_variance_sigma_ma DOUBLE PRECISION NOT NULL,
    power_mw DOUBLE PRECISION NOT NULL,
    window_samples INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT power_readings_topic_chk CHECK (topic ~ '^pylon/[0-9]{2}/power$'),
    CONSTRAINT power_readings_window_samples_chk CHECK (window_samples > 0)
);

SELECT create_hypertable(
    'power_readings',
    by_range('event_time', INTERVAL '1 day'),
    if_not_exists => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_power_readings_dedup
    ON power_readings (event_time, node_id, topic);
CREATE INDEX IF NOT EXISTS idx_power_node_time
    ON power_readings (node_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_power_attack_time
    ON power_readings (attack_state, event_time DESC);

-- -------------------------------------------------------------------
-- Environment telemetry hypertable
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS env_readings (
    event_time TIMESTAMPTZ NOT NULL,
    topic TEXT NOT NULL,
    node_id SMALLINT NOT NULL REFERENCES devices(node_id),
    device_name TEXT NOT NULL,
    temperature_c DOUBLE PRECISION NOT NULL,
    humidity_pct DOUBLE PRECISION NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT env_readings_topic_chk CHECK (topic ~ '^pylon/[0-9]{2}/env$')
);

SELECT create_hypertable(
    'env_readings',
    by_range('event_time', INTERVAL '1 day'),
    if_not_exists => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_env_readings_dedup
    ON env_readings (event_time, node_id, topic);
CREATE INDEX IF NOT EXISTS idx_env_node_time
    ON env_readings (node_id, event_time DESC);

-- -------------------------------------------------------------------
-- Vibration telemetry hypertable
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vibration_readings (
    event_time TIMESTAMPTZ NOT NULL,
    topic TEXT NOT NULL,
    node_id SMALLINT NOT NULL REFERENCES devices(node_id),
    device_name TEXT NOT NULL,
    accel_x_ms2 DOUBLE PRECISION NOT NULL,
    accel_y_ms2 DOUBLE PRECISION NOT NULL,
    accel_z_ms2 DOUBLE PRECISION NOT NULL,
    gyro_x_rads DOUBLE PRECISION NOT NULL,
    gyro_y_rads DOUBLE PRECISION NOT NULL,
    gyro_z_rads DOUBLE PRECISION NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT vibration_readings_topic_chk CHECK (topic ~ '^pylon/[0-9]{2}/vibration$')
);

SELECT create_hypertable(
    'vibration_readings',
    by_range('event_time', INTERVAL '1 day'),
    if_not_exists => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_vibration_readings_dedup
    ON vibration_readings (event_time, node_id, topic);
CREATE INDEX IF NOT EXISTS idx_vibration_node_time
    ON vibration_readings (node_id, event_time DESC);

-- -------------------------------------------------------------------
-- Convenience view for LA-DT ML feature extraction
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW ml_power_features AS
SELECT
    p.event_time,
    p.node_id,
    p.device_name,
    p.attack_state,
    p.sensor_voltage_mean_v,
    p.sensor_voltage_peak_v,
    p.absolute_current_ma,
    p.variance_current_ma,
    p.peak_current_ma,
    p.current_variance_sigma_ma,
    p.power_mw,
    p.window_samples
FROM power_readings p;
