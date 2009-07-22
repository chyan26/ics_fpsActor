import logging
import os

from twisted.internet import reactor

import opscore.actor.model as opsModel
import actorcore.Actor as coreActor
import actorcore.CmdrConnection as coreCmdr

import pdb

class Toy(coreActor.Actor):
    def __init__(self, name, productName=None, configFile=None, debugLevel=30):
        coreActor.Actor.__init__(self, name, productName=productName, configFile=configFile)

        self.logger.setLevel(debugLevel)

        self.cmdr = coreCmdr.Cmdr('%s.%s' % (name, name), self)
        self.cmdr.connectionMade = self.connectionMade
        self.cmdr.connect()

        self.run()

    def connectionMade(self):
        self.bcast.warn("Toy is connected.")
        
def test1():
    toy = Toy('toy', productName='toyActor', debugLevel=5)
    
if __name__ == "__main__":
    test1()
