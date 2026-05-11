#define USER_ID           1234   // 0–65535
#define APPLICATION_ID    5678   // 0–65535

#define SEND_INTERVAL_MS  3000

void sendNfcData(uint16_t userId, uint16_t appId) {
  char payload[5];
  payload[0] = (char)((userId >> 8) & 0xFF);   // high byte of user_id
  payload[1] = (char)(userId & 0xFF);           // low  byte of user_id
  payload[2] = (char)((appId >> 8) & 0xFF);    // high byte of app_id
  payload[3] = (char)(appId & 0xFF);            // low  byte of app_id
  payload[4] = '\n';                            // terminator Python splits on

  Serial.write((uint8_t*)payload, 5);
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  delay(100);
}

void loop() {
  sendNfcData(0000, 0000);
  delay(SEND_INTERVAL_MS);
  sendNfcData(9000, 9000);
  delay(SEND_INTERVAL_MS);
}