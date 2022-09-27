#!/usr/bin/env python

import actorcore.ICC
from twisted.internet import reactor

class Visitor(object):
    def __init__(self, actor):
        """Keeper of persistent visit/frame/cmd state.

        FPS commands update visit and cmd.
        Camera/cobraCoach consume frame numbers.
        """
        self.actor = actor

        self.visit = None
        self.frameSeq = 0
        self.cmd = self.actor.bcast

    def getNextFrameNum(self):
        """Get the next full (visit}{frameSeq}. Does increment self.frameSeq. """

        frameSeq = self.frameSeq
        self.frameSeq += 1
        if frameSeq >= 100:
            raise ValueError(f"frameSeq > 100 (would be {self.frameSeq})")

        frameNum = self.visit * 100 + frameSeq
        return frameNum

    def setOrGetVisit(self, cmd):
        """Set and return the visit passed in the command keys, or fetch one from gen2. """

        self.cmd = cmd
        cmdKeys = cmd.cmd.keywords

        # When we start a new visit, always reset frame counter.
        newVisit = 'visit' not in cmdKeys or cmdKeys['visit'].values[0] != self.visit
        if newVisit:
            self.frameSeq = 0

        if 'visit' in cmdKeys:
            self.visit = cmdKeys['visit'].values[0]
        else:
            ret = self.actor.cmdr.call(actor='gen2', cmdStr='getVisit caller=fps',
                                       forUserCmd=cmd, timeLim=15.0)
            if ret.didFail:
                raise RuntimeError("getNextFilename failed getting a visit number in 15s!")
            self.visit = self.actor.models['gen2'].keyVarDict['visit'].valueList[0]

        return self.visit


class Fps(actorcore.ICC.ICC):
    def __init__(self, name, productName=None, debugLevel=30):
        # This sets up the connections to/from the hub, the logger, and the twisted reactor.
        #
        self.db = None
        self.cc = None

        actorcore.ICC.ICC.__init__(self, name, productName=productName)

        self.everConnected = False

    def connectionMade(self):
        if self.everConnected is False:
            self.everConnected = True
            self.bcast.inform('text="connection made!"')

            _needModels = ('mcs', 'fps', 'gen2')
            self.logger.info(f'adding models: {_needModels}')
            self.addModels(_needModels)
            self.logger.info(f'added models: {self.models.keys()}')

            self.visitor = Visitor(self)

            reactor.callLater(2, self.callCommand, 'loadModel')

    def getPositionsForFrame(self, frameId):
        return self.cmdSets['FpsCmd'].getPositionsForFrame(frameId)


def main():
    fps = Fps('fps', 'fpsActor')
    fps.run()


if __name__ == '__main__':
    main()
