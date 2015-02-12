#!/usr/bin/env python

import actorcore.Actor

class Pfics(actorcore.Actor.Actor):
    def __init__(self, name, productName=None, configFile=None, debugLevel=30):
        # This sets up the connections to/from the hub, the logger, and the twisted reactor.
        #
        actorcore.Actor.Actor.__init__(self, name, productName=productName, configFile=configFile,
                                       modelNames=('mcs','mps'))
        
def main():
    pfics = Pfics('pfics', 'pficsActor')
    pfics.run()

if __name__ == '__main__':
    main()
    
