import logging
import os

from twisted.internet import reactor

import opscore.actor.model as opsModel
import actorcore.Actor as coreActor
import actorcore.CmdrConnection as coreCmdr

class Toy(coreActor.Actor):
    def __init__(self, name, debugLevel=30):
        coreActor.Actor.__init__(self, name, configFile)

        self.logger.setLevel(debugLevel)

        self.cmdr = coreCmdr.Cmdr('%s.%s' % (name, name), self)
        self.cmdr.connectionMade = self.connectionMade
        self.cmdr.connect()

        self.run()

    def connectionMade(self):
        self.bcast.warn("Toy is connected.")
        
def test1():
    toy = Toy('name', os.path.expandvars('$SOP_DIR/etc/toy.cfg'),
              debugLevel=5)
    
if __name__ == "__main__":
    test1()
