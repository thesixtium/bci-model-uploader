#include <Wire.h>
#include <SPI.h>
#include <Adafruit_PN532.h>

#define PN532_SS (10)

Adafruit_PN532 nfc(PN532_SS);

// ── Configuration ─────────────────────────────────────────────
#define USER_DATA_PAGE   4      // First writable user page on NTAG215
#define USER_DATA_MARKER 0xAB   // Magic byte so we know the tag is ours

// Layout of page 4:
//   byte 0-1 : userId  (big-endian uint16_t)
//   byte 2-3 : appId   (big-endian uint16_t)
// ──────────────────────────────────────────────────────────────

// Write userId + appId to page 4 of an NTAG215.
// Returns true on success.
bool writeUserData(uint16_t userId, uint16_t appId) {
  uint8_t page[4];

  // Pack big-endian
  page[0] = (userId >> 8) & 0xFF;
  page[1] =  userId       & 0xFF;
  page[2] = (appId  >> 8) & 0xFF;
  page[3] =  appId        & 0xFF;

  Serial.print("Writing page ");
  Serial.print(USER_DATA_PAGE);
  Serial.print(": ");
  nfc.PrintHex(page, 4);

  bool ok = nfc.ntag2xx_WritePage(USER_DATA_PAGE, page);
  if (ok) {
    Serial.println("Write successful.");
  } else {
    Serial.println("Write FAILED.");
  }
  return ok;
}

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

// ── Setup ─────────────────────────────────────────────────────
void setup(void) {
  Serial.begin(115200);
  while (!Serial) delay(10);

  nfc.begin();

  uint32_t versiondata = nfc.getFirmwareVersion();
  if (!versiondata) {
    Serial.println("Didn't find PN53x board");
    while (1);
  }

  Serial.print("Found chip PN5"); Serial.println((versiondata >> 24) & 0xFF, HEX);
  Serial.print("Firmware ver. "); Serial.print((versiondata >> 16) & 0xFF, DEC);
  Serial.print('.'); Serial.println((versiondata >> 8) & 0xFF, DEC);

  Serial.println("Waiting for an NTAG215 card...");
}

// ── Loop ──────────────────────────────────────────────────────
void loop(void) {
  uint8_t uid[7];
  uint8_t uidLength;

  bool found = nfc.readPassiveTargetID(PN532_MIFARE_ISO14443A, uid, &uidLength);

  if (!found) return;

  Serial.println("\nFound ISO14443A card");
  Serial.print("  UID Length: "); Serial.print(uidLength); Serial.println(" bytes");
  Serial.print("  UID Value: "); nfc.PrintHex(uid, uidLength);

  if (uidLength != 7) {
    Serial.println("Not an NTAG2xx tag — skipping.");
    return;
  }

  // ── Choose: write or read ────────────────────────────────────
  // Change these values to whatever you want to store, then
  // set WRITE_MODE to true for one scan, then back to false.
  const bool     WRITE_MODE = true;
  const uint16_t MY_USER_ID = 0x0000;
  const uint16_t MY_APP_ID  = 0x0001;
  // ─────────────────────────────────────────────────────────────

  if (WRITE_MODE) {
    Serial.println("\n-- WRITE MODE --");
    writeUserData(MY_USER_ID, MY_APP_ID);
  }

  // Always read back and print after any operation
  Serial.println("\n-- READ BACK --");
  uint16_t userId, appId;
  if (readUserData(userId, appId)) {
    Serial.print("  userId : 0x"); Serial.println(userId, HEX);
    Serial.print("  appId  : 0x"); Serial.println(appId,  HEX);
  }

  Serial.println("\nSend any character to scan another tag.");
  Serial.flush();
  while (!Serial.available());
  while (Serial.available()) Serial.read();
}