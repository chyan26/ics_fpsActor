#!/usr/bin/env python

import time

import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr

class PficsCmd(object):

    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor

        # Declare the commands we implement. When the actor is started
        # these are registered with the parser, which will call the
        # associated methods when matched. The callbacks will be
        # passed a single argument, the parsed and typed command.
        #
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('setupField', 'fieldID', self.setupField),
            ('loop', '<cnt> [<expTime>]', self.loop),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("pfics_pfics", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("fieldID", types.String(), 
                                                 help="ID for the field, which defines the fiber positions"),
                                        keys.Key("expTime", types.Float(), 
                                                 help="Seconds for exposure"))

    def ping(self, cmd):
        """Query the actor for liveness/happiness."""

        # 
        cmd.finish("text='Present and (probably) well'")

    def status(self, cmd):
        """Report status and version; obtain and send current data"""

        self.actor.sendVersionKey(cmd)

        keyStrings = ['text="nothing to say, really"']
        keyMsg = '; '.join(keyStrings)

        cmd.inform(keyMsg)
        cmd.diag('text="still nothing to say"')
        cmd.finish()

    def setupField(self, cmd):
        """ Fully configure all the fibers for the given field. """

        # 
        cmd.fail("text='Not yet implemented'")

    def loop(self, cmd):
        """ Run the expose-move loop a few times. For developement. """

        cnt = cmd.cmd.keywords["cnt"].values[0]
        expTime = cmd.cmd.keywords["expTime"].values[0] if "expTime" in cmd.cmd.keywords else 0.5
        
        for i in range(cnt):
            cmdString = "expose expTime=%0.1f" % (expTime)
            cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                          forUserCmd=cmd, timeLim=expTime+5.0)
            if cmdVar.didFail:
                cmd.fail('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
                return
            
        cmd.finish()

