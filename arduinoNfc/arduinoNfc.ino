#include <Wire.h>
#include <SPI.h>
#include <Adafruit_PN532.h>

#define PN532_SS (10)
#define USER_DATA_PAGE   4      // First writable user page on NTAG215
#define USER_DATA_MARKER 0xAB   // Magic byte so we know the tag is ours

Adafruit_PN532 nfc(PN532_SS);

// Read userId + appId back from page 4.
// Returns true on success and fills the out-parameters.
bool readUserData(uint16_t &userId, uint16_t &appId) {
  uint8_t data[4];

  if (!nfc.ntag2xx_ReadPage(USER_DATA_PAGE, data)) {
    Serial.println("Read failed.");
    return false;
  }

  // Unpack big-endian
  userId = ((uint16_t)data[0] << 8) | data[1];
  appId  = ((uint16_t)data[2] << 8) | data[3];
  return true;
}

void sendNfcData(uint16_t userId, uint16_t appId) {
  char payload[5];
  payload[0] = (char)((userId >> 8) & 0xFF);   // high byte of user_id
  payload[1] = (char)(userId & 0xFF);           // low  byte of user_id
  payload[2] = (char)((appId >> 8) & 0xFF);    // high byte of app_id
  payload[3] = (char)(appId & 0xFF);            // low  byte of app_id
  payload[4] = '\n';                            // terminator Python splits on

  Serial.write((uint8_t*)payload, 5);
}

void setup(void) {
  Serial.begin(115200);
  while (!Serial) delay(10);

  nfc.begin();

  uint32_t versiondata = nfc.getFirmwareVersion();
  if (!versiondata) {
    while (1);
  }
}

void loop(void) {
  uint8_t uid[7];
  uint8_t uidLength;

  bool found = nfc.readPassiveTargetID(PN532_MIFARE_ISO14443A, uid, &uidLength);

  if (!found) return;

  if (uidLength != 7) {
    Serial.println("Not an NTAG2xx tag — skipping.");
    return;
  }

  uint16_t userId, appId;
  if (readUserData(userId, appId)) {
    sendNfcData( userId, appId);
    delay(1000);
  }
}