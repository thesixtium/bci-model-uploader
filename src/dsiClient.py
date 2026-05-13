import subprocess
from os.path import join
from pylsl import StreamInlet, resolve_byprop

DSI_PORT = "COM3"
DSI_STREAMER_NAME = "DSI-Streamer-v.1.08.119.exe"

def run_dsi_streamer():
    # DSI Streamer (for headset impedance checking)
    try:
        dsi_streamer_path = join("DSI_Streamer", "DSI-Streamer-v.1.08.119.exe")
        subprocess.run([dsi_streamer_path], check = True)
    except subprocess.CalledProcessError as e:
        print(f"DSI Streamer failed with error code {e.returncode}")
        raise Exception(f"Error running DSI Streamer: {e}")
    print("DSI Streamer closed")

def run_dsi2lsl():
    # DSI2LSL
    try:
        dsi2lsl_path = join("dsi2lsl", "dsi2lsl.exe")
        subprocess.run(
            [
                dsi2lsl_path,
                f'port={DSI_PORT}',
                'lsl-stream-name=DSI7',
                'montage=F4,C4,S1,S3,C3,F3'
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"DSI2LSL failed with error code {e.returncode}")
        raise Exception(f"Error running DSI2LSL: {e}")

def main():
    # start DSI Streamer
    #try:
     #   run_dsi_streamer()
    #except Exception as e:
     #   error_msg = f"DSI Streamer failed: {e}"
      #  print(error_msg)

    # start DSI2LSL
    #try:
       # run_dsi2lsl()
    #except Exception as e:
     #   error_msg = f"DSI2LSL failed: {e}"
      #  print(error_msg)

    print("looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG")

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        chunk, timestamps = inlet.pull_chunk()
        if timestamps:
            print(timestamps, chunk)

if __name__ == "__main__":
    main()