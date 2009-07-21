#!/usr/bin/env python

""" DebugCmd.py -- wrap debugging functions. """

import logging
import Commands.CmdSet
from opscore.utility.qstr import qstr
import time

class DebugCmd(Commands.CmdSet.CmdSet):
    """ Wrap debugging commands.  """
    
    def __init__(self, icc):
        Commands.CmdSet.CmdSet.__init__(self, icc)
        
        self.help = (('call', 'send a command elsewhere, just for fun.'),)
        self.vocab = {
            'call':self.call_cmd,
            }

    def call_cmd(self, cmd):
        """ Send a command to a different actor. """

        try:
            cmdArgs = cmd.cmd.keywords
            actor = cmdArgs['actor'].values[0]
            cmdStr = cmdArgs['cmd'].values[0]
        except Exception, e:
            cmd.fail('text="actor and cmd must be specified"')
            return
        
        if 'timeLim' in cmdArgs:
            timeLim = float(cmdArgs['timeLim'].values[0])
        else:
            timeLim = 0

        if 'cnt' in cmdArgs:
            cnt = float(cmdArgs['cnt'].values[0])
        else:
            cnt = 1

        if 'delay' in cmdArgs:
            delay = float(cmdArgs['delay'].values[0])
        else:
            delay = 0
            
        for i in range(cnt):
            cmdvar = self.icc.cmdr.call(actor=actor, cmdStr=cmdStr, timeLim=timeLim)
            cmd.respond('text="done with iteration %d of %d: %s"' % (i+1, cnt, cmdvar.replyList))

            if delay:
                time.sleep(delay)
                
        cmd.finish()
        
