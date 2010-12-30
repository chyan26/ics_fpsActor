#!/usr/bin/env python

import pdb
import logging
import pprint
import re, sys, time
from time import sleep

import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr
import actorcore.help as help

class ToyCmd(object):

	def __init__(self, actor):
		self.actor = actor

		# Define some typed command arguemetnts
		self.keys = keys.KeysDictionary("toy_toy", (1, 1),
										keys.Key("cartridge", types.Int(), help="A cartridge ID"),
										keys.Key("actor", types.String(), help="Another actor to command"),
										keys.Key("cmd", types.String(), help="A command string"),
										keys.Key("count", types.Int(), help="A count of things to do"))
		#
		# Declare commands
		#
		self.vocab = [
			('status', '', self.status),
			('update', '', self.update),
			('doSomething', '<count>', self.doSomething),
			('passAlong', 'actor <cmd>', self.passAlong),
		]

	def ping(self, cmd):
		'''Query the actor for liveness/happiness.'''

		cmd.finish("text='Present and (probably) well'")

	def status(self, cmd):
		'''Report status and version; obtain and send current data'''

		self.actor.sendVersionKey(cmd)
		self.doStatus(cmd, flushCache=True)

	def update(self, cmd):
		'''Report status and version; obtain and send current data'''

		self.doStatus(cmd=cmd)

	def doStatus(self, cmd=None, flushCache=False, doFinish=True):
		'''Report full status'''

		if not cmd:
			cmd = self.actor.bcast

		keyStrings = ['text="nothing to say, really"']
		keyMsg = '; '.join(keyStrings)

		cmd.inform(keyMsg)
		cmd.diag('text="still nothing to say"')
		cmd.finish()

	def doSomething(self, cmd):
		""" Do something pointless. """

        cnt = cmd.cmd.keywords["cnt"].values[0]
		for i in range(cnt):
			self.respond('cnt=%d' % (i))
		self.finish()

	def passAlong(self, cmd):
		""" Pass a command along to another actor. """

        actor = cmd.cmd.keywords["actor"].values[0]
		cmdString = cmd.cmd.keywords["cmd"].values[0]

		cmdVar = self.actor.cmdr.call(actor=actor, cmdStr=cmdString, timeLim=30.0)
		if cmdVar.didFail:
			cmd.fail('text=%s' % (qstr('Failed to pass %s along to %s' % (cmdStr, actor))))
		else:
			self.finish()

	def sendingCommands(self, cmd):
		""" Examples of sending commands to other actors. """
		
		cmdVar = self.actor.cmdr.call(actor='tcc', cmdStr='axis status', timeLim=3.0)
		if cmdVar.didFail:
			cmd.warn('text=\'Failed to fetch TCC axis status\'')

