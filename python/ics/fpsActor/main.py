#!/usr/bin/env python

import opscore.utility.sdss3logging
import actorcore.ICC

class Fps(actorcore.ICC.ICC):
    def __init__(self, name, productName=None, configFile=None, debugLevel=30):
        # This sets up the connections to/from the hub, the logger, and the twisted reactor.
        #
        super().__init__(name, 
                         productName=productName,
                         modelNames=('gen2'),
                         configFile=configFile)

        self.everConnected = False

    def connectionMade(self):
        if self.everConnected is False:
            self.everConnected = True
            self.bcast.inform('text="connection made!"')

            _needModels=('mcs','fps','gen2')
            self.logger.info(f'adding models: {_needModels}')
            self.addModels(_needModels)
            self.logger.info(f'added models: {self.models.keys()}')

            # reactor.callLater(10, self.status_check)

    def getPositionsForFrame(self, frameId):
        return self.cmdSets['FpsCmd'].getPositionsForFrame(frameId)

def main():
    fps = Fps('fps', productName='fpsActor')
    fps.run()

if __name__ == '__main__':
    main()
    
