from bppy import *

def starts_with_a(e):
    return e.startswith("A")

def starts_with_c(e):
    return e.startswith("C")

@b_thread
def screen_1():
    yield {request: BEvent("L1")}
    yield {request: BEvent("A1")}

@b_thread
def screen_2():
    yield {request: BEvent("L2")}
    yield {request: BEvent("C2")}

@b_thread
def add_before_checkout():
    yield {waitFor: EventSet(lambda e: e.name.startswith("A")), block: EventSet(lambda e: e.name.startswith("C"))}


b_program = BProgram(bthreads=[screen_1(), screen_2(), add_before_checkout()],
                     event_selection_strategy=SimpleEventSelectionStrategy(),
                     listener=PrintBProgramRunnerListener())

b_program.run()
