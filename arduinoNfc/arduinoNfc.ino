void setup() {
  Serial.begin(9600);
}

void loop() {
  Serial.write( "!model1?" );
  delay( 10000 );
  Serial.write( "!model2?" );
  delay( 10000 );
  Serial.write( "!model2?" );
  delay( 10000 );
}
