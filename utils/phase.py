class Phase:
    '''
    A scheduler that manages the phase with a Queue
    '''
    def __init__(self, max_phase, equilibrium_max_phase):
        self.max_phase = max_phase
        self.equilibrium_max_phase = equilibrium_max_phase
        self.phase = 0
        self.equilibrium_phase = 0
        self.equilibrium_mode = False

    def __repr__(self):
        return 'phase: {}, equilibrium_phase: {}'.format(
            self.phase, self.equilibrium_phase
        )

    def __str__(self):
        return str(self.phase)

    def __add__(self, other):
        if self.equilibrium_mode:
            self.equilibrium_phase += other
        self.phase += other
        return self

    def set_complete(self):
        if not self.equilibrium_mode:
            self.equilibrium_phase += 1
        self.equilibrium_mode = True

    @property
    def finished(self):
        return (self.phase > self.max_phase) \
               or (self.equilibrium_phase > self.equilibrium_max_phase)
