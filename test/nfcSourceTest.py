from src.nfcSource import NfcSource
import time

from src.classifierRunner import ClassifierRunner
from src.eegSource import EegSource


if __name__ == '__main__':
    cs = NfcSource()
    es = EegSource()
    cr = ClassifierRunner( cs, es, None, None )

    cs.setup()
    cr.run()

    try:
        while True:
            time.sleep( 0.1 )
    except KeyboardInterrupt:
        print( "Stopping..." )
        cs.stop()
        exit()