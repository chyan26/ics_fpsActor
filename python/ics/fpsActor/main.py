#!/usr/bin/env python

import actorcore.ICC

class Fps(actorcore.ICC.ICC):
    def __init__(self, name, productName=None, debugLevel=30):
        # This sets up the connections to/from the hub, the logger, and the twisted reactor.
        #
        actorcore.Actor.Actor.__init__(self, name, productName=productName,
                                       modelNames=('mcs','fps','gen2'))

def main():
    fps = Fps('fps', 'fpsActor')
    fps.run()

if __name__ == '__main__':
    main()
    
