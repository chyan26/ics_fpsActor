#!/usr/bin/env python

import opscore.utility.sdss3logging

import opscore.actor.model
import actorcore.Actor

class Pfics(actorcore.Actor.Actor):
    def __init__(self, name, productName=None, configFile=None, debugLevel=30):
        # This sets up the connections to/from the hub, the logger, and the twisted reactor.
        #
        actorcore.Actor.Actor.__init__(self, name, productName=productName, configFile=configFile)
        
        # Explicitly declare which actors we need to know about, so we can access their keywords.
        #
        self.models = {}
        for actor in ['mcs']:
            self.models[actor] = opscore.actor.model.Model(actor)

#
# To work

if __name__ == '__main__':
    pfics = Pfics('pfics', 'pficsActor')
    pfics.run()
    
