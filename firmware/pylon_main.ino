/*
 * PowerDetect Unified Firmware — All 7 attack types
 * Flash on all 5 ESP32 nodes (change CURRENT_NODE_ID only)
 * Core 0: Deterministic 500 Hz ADC + Welford DSP
 * Core 1: Sensors, MQTT, attack tasks
 */
#include <ArduinoJson.h>
#include <DHT.h>
#include <MPU6500_WE.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <esp_adc_cal.h>
#include <esp_timer.h>
#include <esp_wifi.h>
#include <math.h>
#include <time.h>

// ─── CHANGE THIS PER NODE (1-5) ─────────────────────────────────────
#define CURRENT_NODE_ID 5

// ─── NETWORK ────────────────────────────────────────────────────────
const char *SSID = "SSID";
const char *PASSWORD = "PASSWORD";
const char *MQTT_SERVER = "[IP_ADDRESS]";
const uint16_t MQTT_PORT = 1883U;

// ─── PINS ───────────────────────────────────────────────────────────
#define DHT_PIN 4
#define DHT_TYPE DHT11
#define SDA_PIN 21
#define SCL_PIN 22
#define ACS712_PIN 34
#define STATUS_LED_PIN 19
#define ATTACK_LED_PIN 18

// ─── CALIBRATION (per-node ACS712 zero/tare voltages) ───────────────
typedef struct {
  float zero_v;
  float tare_v;
} NodeCalibration;
static const NodeCalibration CAL[6] = {
    {0.0F, 0.0F},           {2.526757F, 2.541690F}, {2.545885F, 2.565905F},
    {2.525538F, 2.542925F}, {2.498624F, 2.517896F}, {2.549931F, 2.568291F}};
static const float SENS = 0.185F;
static const float VDR = 14.7F / 10.0F;
static const float VSUP = 5.0F;

// ─── TIMING ─────────────────────────────────────────────────────────
static const uint32_t ENV_MS = 10000UL;
static const uint32_t VIB_MS = 10000UL;
static const uint32_t PWR_MS = 2000UL;
static const uint32_t DSP_US = 2000UL;

// ─── MQTT TOPICS (auto-generated) ───────────────────────────────────
static char DEV_NAME[32];
static char T_ENV[64], T_VIB[64], T_PWR[64], T_CTL[64];

// ─── GLOBALS ────────────────────────────────────────────────────────
static DHT dht(DHT_PIN, DHT_TYPE);
static MPU6500_WE mpu = MPU6500_WE(0x68);
static WiFiClient espClient;
static PubSubClient client(espClient);
static WiFiUDP udp;
static esp_adc_cal_characteristics_t adc_chars;

// ─── DSP STATE (Core 0, atomically accessed) ────────────────────────
typedef struct {
  uint64_t n;
  double mean;
  double m2;
  uint32_t peak;
} DspAcc;
static portMUX_TYPE dspMux = portMUX_INITIALIZER_UNLOCKED;
static volatile DspAcc dsp = {0, 0.0, 0.0, 0};
static TaskHandle_t adcTask = NULL;
static esp_timer_handle_t adcTimer = NULL;

// ─── ATTACK STATE ───────────────────────────────────────────────────
static String atk_type = "none";
static uint32_t atk_end = 0UL;
static TaskHandle_t hDDoS = NULL;
static TaskHandle_t hKami = NULL;
static TaskHandle_t hExh = NULL;
static TaskHandle_t hTamp = NULL;
static volatile bool kami_active = false;
static String kami_flush = "none";
static volatile bool disc_active = false;
static String disc_flush = "none";
static bool spoof_dht = false;
static bool spoof_mpu = false;
static uint32_t t_env = 0, t_vib = 0, t_pwr = 0;

// ─── CORE 0: ADC TIMER + TASK ───────────────────────────────────────
static void IRAM_ATTR adcCB(void *) {
  if (adcTask) {
    BaseType_t w = pdFALSE;
    vTaskNotifyGiveFromISR(adcTask, &w);
    if (w)
      portYIELD_FROM_ISR();
  }
}
static void adcTaskFn(void *) {
  while (true) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    uint32_t mv = esp_adc_cal_raw_to_voltage((uint32_t)analogRead(ACS712_PIN),
                                             &adc_chars);
    portENTER_CRITICAL(&dspMux);
    dsp.n++;
    if (mv > dsp.peak)
      dsp.peak = mv;
    double x = (double)mv, d = x - dsp.mean;
    dsp.mean += d / (double)dsp.n;
    dsp.m2 += d * (x - dsp.mean);
    portEXIT_CRITICAL(&dspMux);
  }
}

// ─── TIME ───────────────────────────────────────────────────────────
static String fmtTime() {
  time_t now = time(NULL);
  struct tm ti;
  char b[25];
  if (!localtime_r(&now, &ti))
    return "1970-01-01 00:00:00";
  strftime(b, sizeof(b), "%Y-%m-%d %H:%M:%S", &ti);
  return String(b);
}

// ─── LED HELPERS ────────────────────────────────────────────────────
static void ledStatus(bool on) {
  digitalWrite(STATUS_LED_PIN, on ? HIGH : LOW);
}
static void ledAttack(bool on) {
  digitalWrite(ATTACK_LED_PIN, on ? HIGH : LOW);
}

// ─── PUBLISH: ENV ───────────────────────────────────────────────────
static void pubEnv(float t, float h) {
  StaticJsonDocument<512> doc;
  doc["topic"] = T_ENV;
  JsonObject d = doc.createNestedObject("device");
  d["name"] = DEV_NAME;
  d["type"] = "sensor";
  doc["timestamp"] = fmtTime();
  JsonObject v = doc.createNestedObject("value");
  JsonObject r1 = v.createNestedObject("temperature");
  r1["value"] = t;
  r1["unit"] = "C";
  JsonObject r2 = v.createNestedObject("humidity");
  r2["value"] = h;
  r2["unit"] = "%";
  String p;
  serializeJson(doc, p);
  client.publish(T_ENV, p.c_str());
}

// ─── PUBLISH: VIBRATION ─────────────────────────────────────────────
static void pubVib(float ax, float ay, float az, float gx, float gy, float gz) {
  StaticJsonDocument<1024> doc;
  doc["topic"] = T_VIB;
  JsonObject d = doc.createNestedObject("device");
  d["name"] = DEV_NAME;
  d["type"] = "sensor";
  doc["timestamp"] = fmtTime();
  JsonObject v = doc.createNestedObject("value");
  const char *k[] = {"accel_x", "accel_y", "accel_z",
                     "gyro_x",  "gyro_y",  "gyro_z"};
  const float vals[] = {ax, ay, az, gx, gy, gz};
  const char *u[] = {"m/s2", "m/s2", "m/s2", "rad/s", "rad/s", "rad/s"};
  for (int i = 0; i < 6; i++) {
    JsonObject r = v.createNestedObject(k[i]);
    r["value"] = vals[i];
    r["unit"] = u[i];
  }
  String p;
  serializeJson(doc, p);
  client.publish(T_VIB, p.c_str());
}

// ─── PUBLISH: POWER ─────────────────────────────────────────────────
static void pubPwr(float sm, float sp, float ac, float vc, float pc, float sc,
                   float pw, uint64_t n, const String &lbl) {
  StaticJsonDocument<1400> doc;
  doc["topic"] = T_PWR;
  JsonObject d = doc.createNestedObject("device");
  d["name"] = DEV_NAME;
  d["type"] = "sensor";
  doc["timestamp"] = fmtTime();
  JsonObject v = doc.createNestedObject("value");
  const char *keys[] = {"sensor_voltage_mean_V",
                        "sensor_voltage_peak_V",
                        "absolute_current_mA",
                        "variance_current_mA",
                        "peak_current_mA",
                        "current_variance_sigma_mA",
                        "power_mW"};
  const float vals[] = {sm, sp, ac, vc, pc, sc, pw};
  const char *units[] = {"V", "V", "mA", "mA", "mA", "mA", "mW"};
  for (int i = 0; i < 7; i++) {
    JsonObject r = v.createNestedObject(keys[i]);
    r["value"] = vals[i];
    r["unit"] = units[i];
  }
  JsonObject ws = v.createNestedObject("window_samples");
  ws["value"] = (uint32_t)n;
  ws["unit"] = "count";
  JsonObject as = v.createNestedObject("attack_state");
  as["value"] = lbl;
  String p;
  serializeJson(doc, p);
  client.publish(T_PWR, p.c_str());
}

// ─── ATTACK TASKS ───────────────────────────────────────────────────
static void ddosTaskFn(void *) {
  WiFi.setSleep(false);
  const IPAddress bcast(192, 168, 16, 226);
  uint8_t dg[1024];
  memset(dg, 0xA5, sizeof(dg));
  udp.begin(9999U);
  while (atk_type == "ddos" && (int32_t)(atk_end - millis()) > 0) {
    uint32_t bs = millis();
    while ((millis() - bs) < 10) {
      udp.beginPacket(bcast, 9999U);
      udp.write(dg, sizeof(dg));
      udp.endPacket();
    }
    vTaskDelay(pdMS_TO_TICKS(40));
  }
  WiFi.setSleep(true);
  atk_type = "none";
  ledAttack(false);
  hDDoS = NULL;
  vTaskDelete(NULL);
}

static void kamiTaskFn(void *) {
  WiFi.setSleep(false);
  const IPAddress bcast(192, 168, 16, 226);
  static uint8_t dg[4096];
  memset(dg, 0xA5, sizeof(dg));
  udp.begin(9999U);
  uint32_t spin;
  while (atk_type == "kamikaze" && (int32_t)(atk_end - millis()) > 0) {
    for (uint8_t i = 0; i < 50; i++) {
      udp.beginPacket(bcast, 9999U);
      udp.write(dg, sizeof(dg));
      udp.endPacket();
    }
    spin = 0;
    while (spin < 100000U) {
      spin++;
      (void)esp_random();
    }
    vTaskDelay(0);
  }
  WiFi.setSleep(true);
  kami_flush = "kamikaze";
  kami_active = false;
  atk_type = "none";
  ledAttack(false);
  hKami = NULL;
  vTaskDelete(NULL);
}

static void exhTaskFn(void *) {
  volatile float x = 0.3451F;
  while (atk_type == "exhaustion" && (int32_t)(atk_end - millis()) > 0) {
    WiFi.setSleep(false);
    uint32_t cs = millis();
    while ((millis() - cs) < 2500) {
      x = sinf(x) * cosf(x) + tanf(x + 0.01F) + (float)esp_random();
      if (x > 1000.0F)
        x = 0.1234F;
    }
    WiFi.setSleep(true);
    vTaskDelay(pdMS_TO_TICKS(random(1000, 3000)));
  }
  WiFi.setSleep(true);
  atk_type = "none";
  ledAttack(false);
  hExh = NULL;
  vTaskDelete(NULL);
}

static void tampTaskFn(void *) {
  volatile float x = 0.5566F;
  const IPAddress bcast(192, 168, 16, 255);
  uint8_t pl[32];
  memset(pl, 0xCD, sizeof(pl));
  udp.begin(8888U);
  while (atk_type == "tampering" && (int32_t)(atk_end - millis()) > 0) {
    WiFi.setSleep(false);
    uint32_t s = millis();
    while ((millis() - s) < 2500) {
      x = sinf(x) * cosf(x) + tanf(x + 0.05F) + (float)esp_random();
      if (x > 1000.0F)
        x = 0.5566F;
    }
    udp.beginPacket(bcast, 8888U);
    udp.write(pl, sizeof(pl));
    udp.endPacket();
    WiFi.setSleep(true);
    vTaskDelay(pdMS_TO_TICKS(random(4000, 8000)));
  }
  WiFi.setSleep(true);
  atk_type = "none";
  ledAttack(false);
  hTamp = NULL;
  vTaskDelete(NULL);
}

// ─── ATTACK TRIGGER ─────────────────────────────────────────────────
static void killTask(TaskHandle_t &h) {
  if (h) {
    vTaskDelete(h);
    h = NULL;
  }
}

static void triggerAttack(const String &type, uint32_t dur) {
  atk_end = millis() + dur * 1000UL;
  atk_type = type;
  spoof_dht = false;
  spoof_mpu = false;
  ledAttack(true);

  if (type == "ddos") {
    killTask(hDDoS);
    xTaskCreatePinnedToCore(ddosTaskFn, "DDoS", 4096, NULL, 1, &hDDoS, 1);
  } else if (type == "kamikaze") {
    kami_active = true;
    kami_flush = "none";
    killTask(hKami);
    xTaskCreatePinnedToCore(kamiTaskFn, "Kami", 8192, NULL, 1, &hKami, 1);
  } else if (type == "exhaustion") {
    killTask(hExh);
    xTaskCreatePinnedToCore(exhTaskFn, "Exh", 4096, NULL, 1, &hExh, 1);
  } else if (type == "tampering") {
    killTask(hTamp);
    xTaskCreatePinnedToCore(tampTaskFn, "Tamp", 4096, NULL, 1, &hTamp, 1);
  } else if (type == "spoofing_dht") {
    spoof_dht = true;
  } else if (type == "spoofing_mpu") {
    spoof_mpu = true;
  } else if (type == "disconnect") {
    disc_active = true;
    disc_flush = "none";
    WiFi.disconnect(true, false);
    WiFi.mode(WIFI_OFF);
  }
}

// ─── MQTT CALLBACK ──────────────────────────────────────────────────
static void mqttCB(char *topic, byte *payload, unsigned int len) {
  StaticJsonDocument<256> doc;
  if (!deserializeJson(doc, payload, len) && doc["cmd"] == "attack") {
    triggerAttack(doc["type"].as<String>(), doc["duration"].as<uint32_t>());
  }
}

static void reconnect() {
  while (!client.connected()) {
    ledStatus(false);
    if (client.connect(DEV_NAME)) {
      ledStatus(true);
      client.subscribe(T_CTL);
    } else {
      delay(5000);
    }
  }
}

// ─── SETUP ──────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  snprintf(DEV_NAME, sizeof(DEV_NAME), "pylon_sensor_%02d", CURRENT_NODE_ID);
  snprintf(T_ENV, sizeof(T_ENV), "pylon/%02d/env", CURRENT_NODE_ID);
  snprintf(T_VIB, sizeof(T_VIB), "pylon/%02d/vibration", CURRENT_NODE_ID);
  snprintf(T_PWR, sizeof(T_PWR), "pylon/%02d/power", CURRENT_NODE_ID);
  snprintf(T_CTL, sizeof(T_CTL), "pylon/%02d/control", CURRENT_NODE_ID);

  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(ATTACK_LED_PIN, OUTPUT);
  ledStatus(false);
  ledAttack(false);

  analogSetAttenuation(ADC_11db);
  analogReadResolution(12);
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 3300,
                           &adc_chars);

  dht.begin();
  Wire.begin(SDA_PIN, SCL_PIN);
  if (mpu.init()) {
    mpu.autoOffsets();
    mpu.enableGyrDLPF();
    mpu.setGyrDLPF(MPU6500_DLPF_6);
    mpu.setAccRange(MPU6500_ACC_RANGE_8G);
    mpu.setGyrRange(MPU6500_GYRO_RANGE_500);
    mpu.enableAccDLPF(true);
    mpu.setAccDLPF(MPU6500_DLPF_6);
  }

  WiFi.setAutoReconnect(false);
  WiFi.begin(SSID, PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  WiFi.setSleep(true);

  // Wait for NTP time sync (no 1970 dates)
  configTime(0L, 0L, "pool.ntp.org", "time.nist.gov");
  Serial.print("Waiting for NTP sync");
  struct tm ti;
  while (true) {
    time_t now = time(NULL);
    if (localtime_r(&now, &ti) && ti.tm_year >= (2024 - 1900))
      break;
    Serial.print(".");
    delay(500);
  }
  Serial.println(" synced!");

  client.setBufferSize(1400U);
  client.setServer(MQTT_SERVER, MQTT_PORT);
  client.setCallback(mqttCB);

  xTaskCreatePinnedToCore(adcTaskFn, "ADC", 4096, NULL, 3, &adcTask, 0);
  esp_timer_create_args_t ta = {};
  ta.callback = &adcCB;
  ta.name = "adc500hz";
#ifdef ESP_TIMER_ISR
  ta.dispatch_method = ESP_TIMER_ISR;
#else
  ta.dispatch_method = ESP_TIMER_TASK;
#endif
  esp_timer_create(&ta, &adcTimer);
  esp_timer_start_periodic(adcTimer, (uint64_t)DSP_US);
}

// ─── LOOP (Core 1) ─────────────────────────────────────────────────
void loop() {
  uint32_t now = millis();

  // ── Disconnect auto-recovery ──
  if (disc_active && (int32_t)(now - atk_end) >= 0) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(SSID, PASSWORD);
    uint32_t ws = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - ws) < 15000UL)
      delay(500);
    disc_flush = "disconnect";
    disc_active = false;
    atk_type = "none";
    ledAttack(false);
  }
  if (disc_active) {
    delay(100);
    return;
  }

  // ── MQTT keep-alive ──
  if (!client.connected())
    reconnect();
  client.loop();

  // ── Attack expiry (for non-task attacks: spoofing) ──
  if (atk_type != "none" && (int32_t)(now - atk_end) >= 0) {
    if (atk_type == "spoofing_dht" || atk_type == "spoofing_mpu") {
      spoof_dht = false;
      spoof_mpu = false;
    }
    atk_type = "none";
    ledAttack(false);
  }

  // ── ENV telemetry (10s) ──
  if (!kami_active && (uint32_t)(now - t_env) >= ENV_MS) {
    t_env = now;
    float tc, hp;
    if (spoof_dht) {
      tc = (float)random(150, 450) / 10.0F;
      hp = (float)random(200, 900) / 10.0F;
    } else {
      tc = dht.readTemperature();
      hp = dht.readHumidity();
    }
    if (!isnan(tc) && !isnan(hp))
      pubEnv(tc, hp);
  }

  // ── VIB telemetry (10s) ──
  if (!kami_active && (uint32_t)(now - t_vib) >= VIB_MS) {
    t_vib = now;
    float ax, ay, az, gx, gy, gz;
    if (spoof_mpu) {
      ax = (float)random(-200, 200) / 100.0F * 9.81F;
      ay = (float)random(-200, 200) / 100.0F * 9.81F;
      az = (float)random(-200, 200) / 100.0F * 9.81F;
      gx = (float)random(-500, 500) / 10.0F * 0.0174532925F;
      gy = (float)random(-500, 500) / 10.0F * 0.0174532925F;
      gz = (float)random(-500, 500) / 10.0F * 0.0174532925F;
    } else {
      xyzFloat a = mpu.getGValues(), g = mpu.getGyrValues();
      ax = a.x * 9.81F;
      ay = a.y * 9.81F;
      az = a.z * 9.81F;
      gx = g.x * 0.0174532925F;
      gy = g.y * 0.0174532925F;
      gz = g.z * 0.0174532925F;
    }
    pubVib(ax, ay, az, gx, gy, gz);
  }

  // ── POWER telemetry (2s) ──
  if ((uint32_t)(now - t_pwr) >= PWR_MS) {
    t_pwr = now;
    if (kami_active)
      return; // DSP keeps accumulating

    DspAcc s;
    portENTER_CRITICAL(&dspMux);
    s = *(DspAcc *)&dsp;
    dsp.n = 0;
    dsp.mean = 0.0;
    dsp.m2 = 0.0;
    dsp.peak = 0;
    portEXIT_CRITICAL(&dspMux);

    if (s.n >= 2) {
      float mv_m = (float)s.mean / 1000.0F, mv_p = (float)s.peak / 1000.0F;
      float sm = mv_m * VDR, sp2 = mv_p * VDR;
      float var_mv2 = (float)(s.m2 / (double)(s.n - 1));
      float sig_v = sqrtf(fmaxf(var_mv2, 0.0F)) / 1000.0F * VDR;
      float zv = CAL[CURRENT_NODE_ID].zero_v, tv = CAL[CURRENT_NODE_ID].tare_v;
      float ac = fabsf((sm - zv) / SENS) * 1000.0F;
      float vc = ((sm - tv) / SENS) * 1000.0F;
      float pc = fabsf((sp2 - zv) / SENS) * 1000.0F;
      float sc = (sig_v / SENS) * 1000.0F;
      float pw = ac * VSUP;

      String lbl = atk_type;
      if (kami_flush == "kamikaze") {
        lbl = "kamikaze";
        kami_flush = "none";
      }
      if (disc_flush == "disconnect") {
        lbl = "disconnect";
        disc_flush = "none";
      }
      pubPwr(sm, sp2, ac, vc, pc, sc, pw, s.n, lbl);
    }
  }
  delay(10);
}
