#!/usr/bin/env python

class StateMachine:
    def __init__(self):
        self.curState = None
        self.handlers = {}

    def add_state(self, state, handler):
        self.handlers[state.upper()] = handler

    def set_current_state(self, state):
        self.curState = state.upper()

    def run_event(self, event):
        if self.curState not in self.handlers.keys():
            return False

        newState = self.handlers[self.curState](event)
        self.curState = newState.upper()

        return True
