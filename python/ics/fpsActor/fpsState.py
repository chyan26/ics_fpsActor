class FpsState(object):
    """ Persistent FPS state, which we do not want reloaded when code is. """

    def __init__(self, commander):
        self.commander = commander
        self.pfsDesignId = None
        self.pfsDesign = None

    def setDesign(self, pfsDesignId, pfsDesign):
        self.pfsDesignId = pfsDesignId
        self.pfsDesign = pfsDesign
