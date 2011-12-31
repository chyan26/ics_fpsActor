#!/usr/bin/env python

from twisted.internet import reactor

import opscore.actor.model
import opscore.actor.keyvar

import actorcore.Actor
import actorkeys

# Import sdss3logging before logging if you want to use it
#
import opscore.utility.sdss3logging as sdss3logging
import logging

class Toy(actorcore.Actor.Actor):
    def __init__(self, name, productName=None, configFile=None, debugLevel=30):
        actorcore.Actor.Actor.__init__(self, name, productName=productName, configFile=configFile)

        self.headURL = '$HeadURL: svn+ssh://sdss3svn@sdss3.org/repo/ops/actors/apoActor/trunk/python/apoActor/apoActor_main.py $'

        self.logger.setLevel(debugLevel)
        self.logger.propagate = True

        #
        # Explicitly load other actor models. We usually need these for FITS headers.
        #
        self.models = {}
        for actor in ["mcp", "guider", "platedb", "tcc"]:
            self.models[actor] = opscore.actor.model.Model(actor)

        # Finish by starting the twisted reactor
        #
        self.run()

    def periodicStatus(self):
        '''Run some command periodically'''

        self.callCommand('status')
        reactor.callLater(int(self.config.get(self.name, 'updateInterval')), self.periodicStatus)

    def connectionMade(self):
        '''Runs this after a connection is made from the hub'''

        # Schedule an update.
        #
        reactor.callLater(3, self.periodicStatus)

#
# To work
#
if __name__ == '__main__':
    toy = Toy('toy', 'toyActor')
